#!/usr/bin/env python
from __future__ import print_function
import re
import json
import ROOT
import sqlite3
import argparse
import subprocess
import multiprocessing
from collections import defaultdict


ROOTPREFIX = "root://cms-xrd-global.cern.ch/"
#ROOTPREFIX = "root://eoscms//eos/cms" # for more local files

parser = argparse.ArgumentParser(description="Collect MEs for given lumisections from DQMIO data and upload to a DQMGUI.")

parser.add_argument('dataset', help='dataset name, like "/StreamHIExpress/HIRun2018A-Express-v1/DQMIO"')
parser.add_argument('-r', '--run', help='Run number of run to process', default=None, type=int)
parser.add_argument('-l', '--lumis', help='JSON file with runs/lumisecitons to process (golden JSON format)', default=None)
parser.add_argument('-u', '--upload', help='Upload files to this GUI, instead of just creating them', default=None)
parser.add_argument('-j', '--njobs', help='Number of threads to read files', type=int, default=1)
parser.add_argument('--limit', help='Only load up to LIMIT files', type=int, default=-1)
args = parser.parse_args()


# we can save a lot of time by only scanning some types, if we know all interesting MEs are of these types.
interesting_types = {
  "TH2Fs",
}

interesting_mes = {
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1",
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2",
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3",
  "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4",
  "PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1",
  "PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_1/TkHMap_NumberValidHits_TECM_W1",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_2/TkHMap_NumberValidHits_TECM_W2",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_3/TkHMap_NumberValidHits_TECM_W3",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_4/TkHMap_NumberValidHits_TECM_W4",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_5/TkHMap_NumberValidHits_TECM_W5",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_6/TkHMap_NumberValidHits_TECM_W6",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_7/TkHMap_NumberValidHits_TECM_W7",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_8/TkHMap_NumberValidHits_TECM_W8",
  "SiStrip/MechanicalView/TEC/MINUS/wheel_9/TkHMap_NumberValidHits_TECM_W9",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_1/TkHMap_NumberValidHits_TECP_W1",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_2/TkHMap_NumberValidHits_TECP_W2",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_3/TkHMap_NumberValidHits_TECP_W3",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_4/TkHMap_NumberValidHits_TECP_W4",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_5/TkHMap_NumberValidHits_TECP_W5",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_6/TkHMap_NumberValidHits_TECP_W6",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_7/TkHMap_NumberValidHits_TECP_W7",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_8/TkHMap_NumberValidHits_TECP_W8",
  "SiStrip/MechanicalView/TEC/PLUS/wheel_9/TkHMap_NumberValidHits_TECP_W9",
  "SiStrip/MechanicalView/TIB/layer_1/TkHMap_NumberValidHits_TIB_L1",
  "SiStrip/MechanicalView/TIB/layer_2/TkHMap_NumberValidHits_TIB_L2",
  "SiStrip/MechanicalView/TIB/layer_3/TkHMap_NumberValidHits_TIB_L3",
  "SiStrip/MechanicalView/TIB/layer_4/TkHMap_NumberValidHits_TIB_L4",
  "SiStrip/MechanicalView/TID/MINUS/wheel_1/TkHMap_NumberValidHits_TIDM_D1",
  "SiStrip/MechanicalView/TID/MINUS/wheel_2/TkHMap_NumberValidHits_TIDM_D2",
  "SiStrip/MechanicalView/TID/MINUS/wheel_3/TkHMap_NumberValidHits_TIDM_D3",
  "SiStrip/MechanicalView/TID/PLUS/wheel_1/TkHMap_NumberValidHits_TIDP_D1",
  "SiStrip/MechanicalView/TID/PLUS/wheel_2/TkHMap_NumberValidHits_TIDP_D2",
  "SiStrip/MechanicalView/TID/PLUS/wheel_3/TkHMap_NumberValidHits_TIDP_D3",
  "SiStrip/MechanicalView/TOB/layer_1/TkHMap_NumberValidHits_TOB_L1",
  "SiStrip/MechanicalView/TOB/layer_2/TkHMap_NumberValidHits_TOB_L2",
  "SiStrip/MechanicalView/TOB/layer_3/TkHMap_NumberValidHits_TOB_L3",
  "SiStrip/MechanicalView/TOB/layer_4/TkHMap_NumberValidHits_TOB_L4",
  "SiStrip/MechanicalView/TOB/layer_5/TkHMap_NumberValidHits_TOB_L5",
  "SiStrip/MechanicalView/TOB/layer_6/TkHMap_NumberValidHits_TOB_L6",
  "EcalBarrel/EBOccupancyTask/EBOT digi occupancy",
  "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE -",
  "EcalEndcap/EEOccupancyTask/EEOT digi occupancy EE +",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z -1 P 1",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z -1 P 2",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z 1 P 1",
  "EcalPreshower/ESOccupancyTask/ES Energy Density Z 1 P 2",
  "Hcal/DigiRunHarvesting/Occupancy/depth/depth1",
  "Hcal/DigiRunHarvesting/Occupancy/depth/depth2",
  "Hcal/DigiRunHarvesting/Occupancy/depth/depth3",
  "Hcal/DigiRunHarvesting/Occupancy/depth/depth4",
  "CSC/CSCOfflineMonitor/Occupancy/hOStripsAndWiresAndCLCT",
  "RPC/AllHits/SummaryHistograms/Occupancy_for_Barrel",
  "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap",
  "DT/02-Segments/Wheel-1/numberOfSegments_W-1",
  "DT/02-Segments/Wheel-2/numberOfSegments_W-2",
  "DT/02-Segments/Wheel0/numberOfSegments_W0",
  "DT/02-Segments/Wheel1/numberOfSegments_W1",
  "DT/02-Segments/Wheel2/numberOfSegments_W2",

  "L1T/L1TObjects/L1TEGamma/timing/egamma_eta_phi_bx_0",
  "L1T/L1TObjects/L1TJet/timing/jet_eta_phi_bx_0",
  "L1T/L1TObjects/L1TMuon/timing/muons_eta_phi_bx_0",
  "L1T/L1TObjects/L1TTau/timing/tau_eta_phi_bx_0",
  "L1T/L1TObjects/L1TEGamma/timing/denominator_egamma",
  "L1T/L1TObjects/L1TJet/timing/denominator_jet",
  "L1T/L1TObjects/L1TMuon/timing/denominator_muons",
  "L1T/L1TObjects/L1TTau/timing/denominator_tau",
}

inf = re.compile("([- \[])inf([,}\]])")
nan = re.compile("([- \[])nan([,}\]])")

def tojson(x):
    rootobj = ROOT.TBufferJSON.ConvertToJSON(x)
    return rootobj

def dasquery(dataset):
    if not dataset.endswith("DQMIO"):
        raise Exception("This tool probably cannot read the dataset you specified. The name should end with DQMIO.")
    dasquery = ["dasgoclient",  "-query=file dataset=%s" % dataset]
    print("Querying das ... %s" % dasquery)
    files = subprocess.check_output(dasquery)
    files = files.splitlines()
    print("Got %d files." % len(files))
    return files


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
           
        # we do the saving in here, concurrently with the reading, to avoid
        # needing to copy/move the TH1's.
        # doing a round-trip via JSON would probably also work, but this seems
        # cleaner. For better structure, one could use Generators...
        # but things need to stay in the same process (from multiprocessing).
        filename = "DQM_V0001_R1%04d%04d__perlumiharvested__perlumi%d_%s_v1__DQMIO.root" % (lumi, endlumi, run, treenames[metype])
        prefix = ["DQMData", "Run 1%04d%04d" % (lumi, endlumi)]
        # we open the file only on the first found ME, to avoid empty files.
        result_file = None
        subsystems = set()

        # inclusive range -- for 0 entries, row is left out
        firstidx, lastidx = idxtree.FirstIndex, idxtree.LastIndex
        metree = getattr(f, treenames[metype])
        metree.SetBranchStatus("*",0)
        metree.SetBranchStatus("FullName",1)

        for x in range(firstidx, lastidx+1):
            metree.GetEntry(x)
            mename = str(metree.FullName)
            if mename in interesting_mes:
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
                gotodir(result_file, prefix + [subsys, "Run summary", "EventInfo"])
                s = ROOT.TObjString("<iRun>i=1%04d%04d</iRun>" % (lumi, endlumi))
                s.Write()
                s = ROOT.TObjString("<iLumiSection>i=%s</iLumiSection>" % run)
                s.Write()
                # we could also set iEvent and runStartTimeStamp if we had values.
            result_file.Close()
            files.append(filename)

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
  

pool = multiprocessing.Pool(processes=args.njobs)
ctr = 0
for outfiles in pool.imap_unordered(harvestfile, files):
#for mes_to_store in map(harvestfile, files):
    ctr += 1
    print("Processed %d files of %d, got %d out files...\r" % (ctr, len(files), len(outfiles)),  end='')
print("\nDone.")
