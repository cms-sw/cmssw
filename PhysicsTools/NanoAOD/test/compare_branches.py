#!/usr/bin/env python3
from ROOT import TFile, TTree,TCut, gROOT,TH1F, TCanvas
import glob, re
from optparse import OptionParser
parser = OptionParser(usage="%prog [options] job1 job2 ...")
parser.add_option("--base", dest="base", default="step2.root", help="name of root file")
parser.add_option("--ref", dest="ref", default="ref/", help="path to the reference files")
parser.add_option("--png", dest="png", default="./", help="path to the plots")
parser.add_option("--onlydiff", dest="diff", default=False, action="store_true", help="print only the histograms with differences")
parser.add_option("--selection", dest="selection", default="", help="a selection of events to draw")
parser.add_option("--branch", dest="branch", default=".*", help="a regexp for selecting branches")
(options, args) = parser.parse_args()

def drawInOrder(t1,h1n, t2,h2n):
    N1 = t1.Draw(f'{vname} >> {h1n}', options.selection, '')
    h1=gROOT.Get(h1n)
    #print(h1)
    h2=TH1F(h2n,'',
            h1.GetNbinsX(),
            h1.GetXaxis().GetXmin(),
            h1.GetXaxis().GetXmax())
    h2=gROOT.Get(h2n)
    #print(h2)
    if t2:
        N2 = t2.Draw(f'{vname} >> {h2n}', options.selection, '')
    #print(f'created histograms {h1.GetName()} and {h2.GetName()}')
    #print(binContents(h1))
    #print(binContents(h2))
    return (h1,h2)

def binContents( h ):
    return [ h.GetBinContent(b) for b in range(-1,h.GetNbinsX()+1)]
def emptyHist( h ):
    return all([ b==0 for b in binContents(h) ])

for job in args:
    fconn = glob.glob(f'{job}*/{options.base}')[0]
    fcon = TFile.Open(fconn)
    con = fcon.Get('Events')
    frefn=glob.glob(f'{options.ref}/{job}*/{options.base}')[0]
    fref = TFile.Open(frefn)
    ref = fref.Get('Events')

    print(f'Comparing branch content from {fconn} and {frefn}')
    print(f'{ref.GetEntries()} events in reference tree and {con.GetEntries()} events in comparison tree')

    ## get outside of the files
    gROOT.cd()
    
    branches = [b.GetName() for b in con.GetListOfBranches()]
    ref_branches = [b.GetName() for b in ref.GetListOfBranches()]
    
    for vname in sorted(set(branches)|set(ref_branches)):
        hrefn = f'ref_{vname}'
        hconn = f'con_{vname}'
        
        if not re.match(options.branch, vname): 
            #print(vname,"not in regexp")
            continue
        
        no_base=no_ref=False
        if not vname in branches:
            print(f'Branch {vname} in not in file to compare')
            no_base=True
        if not vname in ref_branches:
            print(f'Branch {vname} in not in file to use as reference')
            no_ref=True

        if no_ref:
            hcon,href=drawInOrder(con, hconn,
                        None, hrefn)
        elif no_base:
            href,hcon=drawInOrder(ref, hrefn,
                        None, hconn)
        else:
            href,hcon=drawInOrder(ref, hrefn,
                        con, hconn)

        href.SetLineColor(1)
        hcon.SetLineColor(2)

        ##ROOT is too stupid for me at this point. If I dont do this, the histograms are empty
        r = sum(binContents(href) + binContents(hcon))

        ##make the difference
        hdiff= TH1F(f'diff_{vname}',
                    '',
                    href.GetNbinsX(),
                    href.GetXaxis().GetXmin(),
                    href.GetXaxis().GetXmax())
    
        hdiff.Add(hcon)
        hdiff.Add(href, -1 )
    
        ymax = max(href.GetMaximum(), hcon.GetMaximum())
        ymin = min(href.GetMinimum(), hcon.GetMinimum(), hdiff.GetMinimum())
    
        hdiff.SetMarkerColor(4)
        hdiff.SetLineColor(4)
        hdiff.SetMarkerStyle(7)


        if options.diff and emptyHist(hdiff):
            #print(f'No changes for branch {vname}')
            continue

        ## show them all
        fig = TCanvas(f'c_{vname}', f'plots for {vname}')
        href.Draw("he")
        hcon.Draw("same he")
        hdiff.Draw("same p e")
        fig.Print(f'{options.png}/{job}_{fig.GetName()}.png')
        fig = TCanvas(f'd_{vname}', f'Difference plot for {vname}')
        hdiff.Draw("hpe")
        fig.Print(f'{options.png}/{job}_{fig.GetName()}.png')
