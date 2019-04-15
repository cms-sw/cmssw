#!/usr/bin/env python
from __future__ import print_function
import re
import json
import ROOT
import sqlite3
import argparse
import subprocess
import multiprocessing


ROOTPREFIX = "root://cms-xrd-global.cern.ch/"
#ROOTPREFIX = "root://eoscms//eos/cms" # for more local files

parser = argparse.ArgumentParser(description="Collect a MEs from DQMIO data, with maximum possible granularity")

parser.add_argument('dataset', help='dataset name, like "/StreamHIExpress/HIRun2018A-Express-v1/DQMIO"')
parser.add_argument('-o', '--output', help='SQLite file to write', default='dqmio.sqlite')
parser.add_argument('-j', '--njobs', help='Number of threads to read files', type=int, default=1)
parser.add_argument('-l', '--limit', help='Only load up to LIMIT files', type=int, default=-1)
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

def tosqlite(x):
    if isinstance(x, ROOT.string):
        try:
            return unicode(x.data())
        except:
            return buffer(x.data())
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x
    if isinstance(x, long):
        return x
    else:
        try: 
            rootobj = unicode(ROOT.TBufferJSON.ConvertToJSON(x))
            # turns out ROOT does not generate valid JSON for NaN/inf
            clean = nan.sub('\\g<1>0\\g<2>', inf.sub('\\g<1>1e38\\g<2>', rootobj))
            obj = json.loads(clean)
            jsonobj = json.dumps(obj, allow_nan=False)
            return jsonobj
        except Exception as e:
            return json.dumps({"root2sqlite_error": e.__repr__(), "root2sqlite_object": x.__repr__()})

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

maketable = """
  CREATE TABLE IF NOT EXISTS monitorelements (
    name,
    fromrun, fromlumi, torun, tolumi,
    metype,
    value
  ); """
makeindex = """
  CREATE INDEX runorder ON monitorelements(fromrun, fromlumi);
"""
insertinto = """
  INSERT INTO monitorelements (
    name,
    fromrun, fromlumi, torun, tolumi,
    metype,
    value
  ) VALUES (
    ?, ?, ?, ?, ?, ?, ?
  ); """
dumpmes = """
  SELECT fromlumi, tolumi, fromrun, name, value FROM monitorelements ORDER BY fromrun, fromlumi ASC;
"""

db = sqlite3.connect(args.output)
db.execute(maketable)
db.execute(makeindex)

def harvestfile(fname):
    f = ROOT.TFile.Open(ROOTPREFIX + fname)
    idxtree = getattr(f, "Indices")
    #idxtree.GetEntry._threaded = True # now the blocking call should release the GIL...

    # we have no good way to find out which lumis where processed in a job.
    # so we watch the per-lumi indices and assume that all mentioned lumis 
    # are covered in the end-of-job MEs. This might fail if there are no 
    # per-lumi MEs.
    knownlumis = set()
    mes_to_store = []

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

                mes_to_store.append((
                  mename,
                  run, lumi, endrun, endlumi,
                  metype,
                  tosqlite(value),
                ))

    return mes_to_store

files = dasquery(args.dataset)
if args.limit > 0: files = files[:args.limit]

pool = multiprocessing.Pool(processes=args.njobs)
ctr = 0
for mes_to_store in pool.imap_unordered(harvestfile, files):
#for mes_to_store in map(harvestfile, files):
    db.executemany(insertinto, mes_to_store);
    db.commit()
    ctr += 1
    print("Processed %d files of %d, got %d MEs...\r" % (ctr, len(files), len(mes_to_store)),  end='')
print("\nDone.")

sqlite2tree = """
// Convert the sqlite format saved above back into a TTree.
// Saving TTrees with objects (TH1's) seems to be close to impossible in Python,
// so we do the roundtrip via SQLite and JSON in a ROOT macro.
// This needs a ROOT with TBufferJSON::FromJSON, which the 6.12 in CMSSW for
// for now does not have. We can load a newer version from SFT (on lxplus6,
// in (!) a cmsenv):
// source /cvmfs/sft.cern.ch/lcg/releases/ROOT/6.16.00-f8770/x86_64-slc6-gcc8-opt/bin/thisroot.sh
// root sqlite2tree.C
// It is rather slow, but the root file is a lot more compact.

int run;
int fromlumi;
int tolumi;
TString* name;
TH2F* value;

int sqlite2tree() {

  auto sql = TSQLiteServer("sqlite:///dev/shm/schneiml/CMSSW_10_5_0_pre1/src/dqmio.sqlite");
  auto query = "SELECT fromlumi, tolumi, fromrun, name, value FROM monitorelements ORDER BY fromrun, fromlumi ASC;";
  auto res = sql.Query(query);

  TFile outfile("/dev/shm/dqmio.root", "RECREATE");
  auto outtree = new TTree("MEs", "MonitorElements by run and lumisection");
  auto nameb     = outtree->Branch("name",    &name);
  auto valueb    = outtree->Branch("value",   &value,128*1024);
  auto runb      = outtree->Branch("run",     &run);
  auto fromlumib = outtree->Branch("fromlumi",&fromlumi);
  auto tolumib   = outtree->Branch("tolumi",  &tolumi);


  while (auto row = res->Next()) {
    fromlumi = atoi(row->GetField(0));
    tolumi   = atoi(row->GetField(1));
    run      = atoi(row->GetField(2));
    name  = new TString(row->GetField(3));
    value = nullptr;
    TBufferJSON::FromJSON(value, row->GetField(4));
    outtree->Fill();
  }
  return 0;
}
"""

    
