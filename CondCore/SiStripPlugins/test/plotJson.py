import json
import ROOT
from pprint import pprint
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="open FILE and extract info", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=False,
                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()

with open(options.filename) as data_file:    
    data = json.load(data_file)
    values = data["data"]
    annotations = data["annotations"]
    title   = annotations["title"]
    x_label = annotations["x_label"]
    y_label = annotations["y_label"]

if(options.verbose):
    pprint(values)

bins=len(values)
print "n. of bins",bins
histo=ROOT.TH1F("histo",title+";"+x_label+";"+y_label,bins,values[0]['x'],values[bins-1]['x'])
for i,value in enumerate(values):
    histo.SetBinContent(i+1,value['y'])

histo.SetLineColor(ROOT.kBlue)
canv=ROOT.TCanvas("c1","c1",800,800)
canv.cd()
histo.Draw()
canv.SaveAs(options.filename.replace(".json",".png"))

    
