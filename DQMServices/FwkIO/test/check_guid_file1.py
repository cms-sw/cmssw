import re
import sys
import ROOT

fname_root = "dqm_file1.root"
fname_report = "dqm_file1_jobreport.xml"

kCmsGuid = "cms::edm::GUID"

f = ROOT.TFile.Open(fname_root)
guid_file = getattr(f, kCmsGuid)
f.Close()

guid_report = None
with open(fname_report) as f:
    guid_re = re.compile("<GUID>(?P<guid>.*)</GUID>")
    for line in f:
        m = guid_re.search(line)
        if m:
            guid_report = m.group("guid")
            break

if guid_report is None:
    print("Did not find GUID from %s" % fname_report)
    sys.exit(1)
if guid_file != guid_report:
    print("GUID from file %s differs from GUID from job report %s" % (guid_file, guid_report))
    sys.exit(1)

print("SUCCEEDED")
