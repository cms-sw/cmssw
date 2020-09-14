################################################
##
## Example usage: python plot2JsonTimeSeries.py --file json_vanilla_short.json --label 'master branch' --file2 json_short.json --label2 'this PR'
##
################################################

from __future__ import print_function
import json
import ROOT
from array import array
from pprint import pprint
from optparse import OptionParser

ROOT.gROOT.SetBatch(True)
ROOT.TGaxis.SetMaxDigits(6);

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="open FILE and extract info", metavar="FILE")

parser.add_option("-l", "--label", dest="label",
                  help="label for the first file", metavar="LABEL")

parser.add_option("-g", "--file2", dest="filename2",
                  help="open FILE 2 and extract info", metavar="FILE2")

parser.add_option("-m", "--label2", dest="label2",
                  help="label for the second file", metavar="LABEL2")

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


with open(options.filename2) as data_file2:    
    data2 = json.load(data_file2)
    values2 = data2["data"]
    annotations2 = data2["annotations"]
    title2   = annotations2["title"]
    x_label2 = annotations2["x_label"]
    y_label2 = annotations2["y_label"]

bins=len(values)
bins2=len(values2)

'''
ahhh the magic of list comprehension
$%#&!@ excepted it does not work!!!

xvalues = [values[i]['x'] for i in range(0,bins)]
yvalues = [values[i]['y'] for i in range(0,bins)]
xvalues2 = [values2[i]['x'] for i in range(0,bins2)]
yvalues2 = [values2[i]['y'] for i in range(0,bins2)]

print(xvalues,yvalues)
print(xvalues2,yvalues2)
'''

xvalues, yvalues = array( 'd' ), array( 'd' )
xvalues2, yvalues2 = array( 'd' ), array( 'd' )

for i in range (bins):
    xvalues.append(values[i]['x'])
    yvalues.append(values[i]['y'])

for i in range (bins2):
    xvalues2.append(values2[i]['x'])
    yvalues2.append(values2[i]['y'])

graph1=ROOT.TGraph(bins,xvalues,yvalues)
graph2=ROOT.TGraph(bins2,xvalues2,yvalues2)

canv=ROOT.TCanvas("c1","c1",1200,800)
canv.SetGrid()
canv.cd()

graph1.SetTitle(title)
graph1.GetXaxis().SetTitle(x_label)
graph1.GetYaxis().SetTitle(y_label)

graph1.SetMarkerSize(1.4)
graph2.SetMarkerSize(1.5)

graph1.SetMarkerStyle(ROOT.kOpenCircle)
graph2.SetMarkerStyle(ROOT.kFullSquare)

graph1.SetMarkerColor(ROOT.kRed)
graph2.SetMarkerColor(ROOT.kBlue)

graph1.SetLineColor(ROOT.kRed)
graph2.SetLineColor(ROOT.kBlue)

graph1.GetYaxis().SetRangeUser(-0.1,5.)

graph1.Draw("APL")
graph2.Draw("PLSame")

TLegend = ROOT.TLegend(0.10,0.80,0.40,0.90)
TLegend.AddEntry(graph1,options.label,"LP")
TLegend.AddEntry(graph2,options.label2,"LP")

TLegend.Draw("same")

canv.Update()
#canv.GetFrame().SetFillColor( 11 )
canv.GetFrame().SetBorderSize( 12 )
canv.Modified()
canv.Update()

outfilename="comparison_"+options.label+"_vs_"+options.label2+"_json.png"
canv.SaveAs(outfilename.replace(" ",""))
