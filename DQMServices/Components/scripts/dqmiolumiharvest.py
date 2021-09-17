#!/usr/bin/env python3
from __future__ import print_function
import os
import json
import ROOT
import fnmatch
import argparse
import subprocess
import multiprocessing
from collections import defaultdict


ROOTPREFIX = "root://cms-xrd-global.cern.ch/"
#ROOTPREFIX = "root://eoscms//eos/cms" # for more local files

parser = argparse.ArgumentParser(description="Collect MEs for given lumisections from DQMIO data and upload to a DQMGUI. " +
                                             "The from-to lumi range will be shown in an artificial run number of form 1xxxxyyyy, while the run number goes into the lumi number field.")

parser.add_argument('dataset', help='dataset name, like "/StreamHIExpress/HIRun2018A-Express-v1/DQMIO"')
parser.add_argument('-r', '--run', help='Run number of run to process', default=None, type=int)
parser.add_argument('-l', '--lumis', help='JSON file with runs/lumisecitons to process (golden JSON format)', default=None)
parser.add_argument('-u', '--upload', help='Upload files to this GUI, instead of just creating them. Delete files after upload.', default=None)
parser.add_argument('-j', '--njobs', help='Number of threads to read files', type=int, default=1)
parser.add_argument('-m', '--me', help='Glob pattern of MEs to load.', default=[], action='append')
parser.add_argument('--limit', help='Only load up to LIMIT files', type=int, default=-1)
parser.add_argument('--perlumionly', help='Only save MEs that cover exactly one lumisection, and use simplified "run" numbers (10xxxx)', action='store_true')
args = parser.parse_args()


# we can save a lot of time by only scanning some types, if we know all interesting MEs are of these types.
interesting_types = {
  "TH2Fs",
  "TH1Fs",
#  "TH2Ds",
#  "TH1Ds",
#  "TH2Ds",
#  "TProfiles",
#  "TProfile2Ds",
}

interesting_mes = args.me
if not interesting_mes:
  print("No --me patterns given. This is fine, but output *will* be empty.")

if args.upload and "https:" in args.upload:
  print("Refuing to upload to production servers, only http upload to local servers allowed.")
  uploadurl = None
else:
  uploadurl = args.upload

def dasquery(dataset):
    if not dataset.endswith("DQMIO"):
        raise Exception("This tool probably cannot read the dataset you specified. The name should end with DQMIO.")
    dasquery = ["dasgoclient",  "-query=file dataset=%s" % dataset]
    print("Querying das ... %s" % dasquery)
    files = subprocess.check_output(dasquery)
    files = files.splitlines()
    print("Got %d files." % len(files))
    return files

files = dasquery(args.dataset)
if args.limit > 0: files = files[:args.limit]

if args.lumis:
  with open(args.lumis) as f:
    j  = json.load(f)
    lumiranges = {int(run): lumis for run, lumis in j.iteritems()}
else:
  if args.run:
    # let's define no lumis -> full run
    lumiranges = {args.run : []}
  else:
    # ... and similarly, no runs -> everything.
    lumiranges = {}

if args.perlumionly:
  perlumionly = True
  def fake_run(lumi, endlumi):
    return "1%05d" % (lumi)
else:
  perlumionly = False
  def fake_run(lumi, endlumi):
    return "1%04d%04d" % (lumi, endlumi)


treenames = { 
  0: "Ints",
  1: "Floats",
  2: "Strings",
  3: "TH1Fs",
  4: "TH1Ss",
  5: "TH1Ds",
  6: "TH2Fs",
  7: "TH2Ss",
  8: "TH2Ds",
  9: "TH3Fs",
  10: "TProfiles",
  11: "TProfile2Ds",
}

def check_interesting(mename):
  for pattern in interesting_mes:
    if fnmatch.fnmatch(mename, pattern):
      return True

def rangecheck(run, lumi):
  if not lumiranges: return True
  if run not in lumiranges: return False
  lumis = lumiranges[run]
  if not lumis: return True
  for start, end in lumis:
    if lumi >= start and lumi <= end:
      return True
  return False

def create_dir(parent_dir, name):
   dir = parent_dir.Get(name)
   if not dir:
      dir = parent_dir.mkdir(name)
   return dir

def gotodir(base, path):
  current = base
  for directory in path[:-1]:
    current = create_dir(current, directory)
    current.cd()


def harvestfile(fname):
    f = ROOT.TFile.Open(ROOTPREFIX + fname)
    idxtree = getattr(f, "Indices")
    #idxtree.GetEntry._threaded = True # now the blocking call should release the GIL...

    # we have no good way to find out which lumis where processed in a job.
    # so we watch the per-lumi indices and assume that all mentioned lumis 
    # are covered in the end-of-job MEs. This might fail if there are no 
    # per-lumi MEs.
    knownlumis = set()
    files = []

    for i in range(idxtree.GetEntries()):
        idxtree.GetEntry(i)
        run, lumi, metype = idxtree.Run, idxtree.Lumi, idxtree.Type
        if lumi != 0:
            knownlumis.add(lumi)

        if not treenames[metype] in interesting_types:
            continue


        endrun = run # assume no multi-run files for now
        if lumi == 0: # per-job ME
            endlumi = max(knownlumis)
            lumi = min(knownlumis)
        else: 
            endlumi = lumi

        if not (rangecheck(run, lumi) or rangecheck(endrun, endlumi)):
          continue
        if perlumionly and lumi != endlumi:
          continue
           
        # we do the saving in here, concurrently with the reading, to avoid
        # needing to copy/move the TH1's.
        # doing a round-trip via JSON would probably also work, but this seems
        # cleaner. For better structure, one could use Generators...
        # but things need to stay in the same process (from multiprocessing).
        filename = "DQM_V0001_R%s__perlumiharvested__perlumi%d_%s_v1__DQMIO.root" % (fake_run(lumi, endlumi), run, treenames[metype])
        prefix = ["DQMData", "Run %s" % fake_run(lumi, endlumi)]
        # we open the file only on the first found ME, to avoid empty files.
        result_file = None
        subsystems = set()

        # inclusive range -- for 0 entries, row is left out
        firstidx, lastidx = idxtree.FirstIndex, idxtree.LastIndex
        metree = getattr(f, treenames[metype])
        # this GetEntry is only to make sure the TTree is initialized correctly
        metree.GetEntry(0)
        metree.SetBranchStatus("*",0)
        metree.SetBranchStatus("FullName",1)

        for x in range(firstidx, lastidx+1):
            metree.GetEntry(x)
            mename = str(metree.FullName)
            if check_interesting(mename):
                metree.GetEntry(x, 1)
                value = metree.Value

                # navigate the TDirectory and save the thing again
                if not result_file:
                    result_file = ROOT.TFile(filename, 'recreate')
                path = mename.split("/")
                filepath = prefix + [path[0], "Run summary"] + path[1:]
                subsystems.add(path[0])
                gotodir(result_file, filepath)
                value.Write()

        # if we found a ME and wrote it to a file, finalize the file here.
        if result_file:
            # DQMGUI wants these to show them in the header bar. The folder name
            # in the TDirectory is also checked and has to match the filename,
            # but the  headerbar can show anything and uses these magic MEs.
            for subsys in subsystems:
                # last item is considerd object name and ignored
                gotodir(result_file, prefix + [subsys, "Run summary", "EventInfo", "blub"])
                s = ROOT.TObjString("<iRun>i=%s</iRun>" % fake_run(lumi, endlumi))
                s.Write()
                s = ROOT.TObjString("<iLumiSection>i=%s</iLumiSection>" % run)
                s.Write()
                # we could also set iEvent and runStartTimeStamp if we had values.
            result_file.Close()
            files.append(filename)

    return files

def uploadfile(filename):
    uploadcommand = ["visDQMUpload.py", uploadurl, filename]
    print("Uploading ... %s" % uploadcommand)
    subprocess.check_call(uploadcommand)

pool = multiprocessing.Pool(processes=args.njobs)
ctr = 0
for outfiles in pool.imap_unordered(harvestfile, files):
#for mes_to_store in map(harvestfile, files):
    if uploadurl:
        for f in outfiles:
            uploadfile(f)
            os.remove(f)
    ctr += 1
    print("Processed %d files of %d, got %d out files...\r" % (ctr, len(files), len(outfiles)),  end='')
print("\nDone.")
