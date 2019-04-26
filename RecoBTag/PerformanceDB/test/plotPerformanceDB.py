#! /usr/bin/env python
#
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2010
#
#____________________________________________________________

from __future__ import print_function
from builtins import range
import sys
import math
import commands
from array import array
from ROOT import *

def main():

    executable = "getbtagPerformance"
    rootFile = "rootFile=btag_performance_CSVM.root"
    payload  = "payload=BTAGCSVM" #"payload=MISTAGCSVM"
    OP = "CSVM"
    flavor   = "flavor=b" #l
    typeSF   = "type=SF"
    typeEff  = "type=eff"
    jetpt    = "pt="
    jeteta   = "eta="

    sp = " "

    ptmin = 25.
    ptmax = 240. #240.
    etamin = 0.
    etamax = 2.4
    ptNbins = 30
    etaNbins = 3

    ptbinwidth = (ptmax - ptmin)/ptNbins
    etabinwidth = (etamax - etamin)/etaNbins
    ptarray = array('d')
    pterrarray = array('d')
    etaarray = array('d')
    etaerrarray = array('d')
    SFarray = array('d')
    SFerrarray = array('d')
    effarray = array('d')
    efferrarray = array('d')
    MCeffarray = array('d')
    MCefferrarray = array('d')    
    SFarray_alleta = array('d')
    SFerrarray_alleta = array('d')
    effarray_alleta = array('d')
    efferrarray_alleta = array('d')
    MCeffarray_alleta = array('d')
    MCefferrarray_alleta = array('d')
        

    for ipt in range(0,ptNbins):
        SF_alleta = 0.
        SFerr_alleta = 0.
        Eff_alleta = 0.
        Efferr_alleta = 0.
        MCEff_alleta = 0.
        MCEfferr_alleta = 0.
                
        apt = ptmin + ipt*ptbinwidth
        ptarray.append( apt )
        pterrarray.append( ptbinwidth*0.5 )
                    
        for ieta in range(0,etaNbins):

            aeta = etamin = ieta*etabinwidth

            print("pt = "+str(apt) + " eta = "+str(aeta))
            tmpjetpt = jetpt+str(apt)
            tmpjeteta = jeteta+str(aeta)
    
            allcmd = executable+ sp +rootFile+ sp +payload+ sp +flavor+ sp +typeSF+ sp +tmpjetpt+ sp +tmpjeteta
            print(allcmd)
            output = commands.getstatusoutput(allcmd)
            print(output[1])
            
            if output[0]!=0:
                print(" Error retrieving data from file DB.")

            aSF = float( output[1].split()[2] )
            aSFerr = float( output[1].split()[4] )

            SFarray.append( aSF )
            SFerrarray.append( aSFerr )
            
            allcmd = executable+ sp +rootFile+ sp +payload+ sp +flavor+ sp +typeEff+ sp +tmpjetpt+ sp +tmpjeteta
            #print allcmd
            output = commands.getstatusoutput(allcmd)
            #print output[1]
            
            if output[0]!=0:
                print(" Error retrieving data from file DB.")
                
            aEff = float( output[1].split()[2] )
            aEfferr = float( output[1].split()[4] )

            effarray.append( aEff )
            efferrarray.append( aEfferr )

            MCeffarray.append( aEff/aSF )
            aMCEfferr = math.sqrt( (aSF*aSF*aEfferr*aEfferr + aEff*aEff*aSFerr*aSFerr)/(aSF*aSF*aSF*aSF) )
            MCefferrarray.append( aMCEfferr )
                        
            SF_alleta += aSF
            SFerr_alleta += aSF*aSF
            Eff_alleta += aEff
            Efferr_alleta += aEfferr*aEfferr
            MCEff_alleta += aEff/aSF
            MCEfferr_alleta += aMCEfferr*aMCEfferr
                                  
        SF_alleta = SF_alleta/etaNbins
        SFerr_alleta = math.sqrt(SFerr_alleta)/etaNbins
        Eff_alleta = Eff_alleta/etaNbins
        Efferr_alleta = math.sqrt(Efferr_alleta)/etaNbins
        MCEff_alleta = MCEff_alleta/etaNbins
        MCEfferr_alleta = math.sqrt(MCEfferr_alleta)/etaNbins
                
        SFarray_alleta.append( SF_alleta )
        SFerrarray_alleta.append( SFerr_alleta )
        effarray_alleta.append( Eff_alleta )
        efferrarray_alleta.append( Efferr_alleta )
        MCeffarray_alleta.append( MCEff_alleta )
        MCefferrarray_alleta.append( MCEfferr_alleta )
                
    histogram = {}
    histogram["SF_jet_pt"] = TGraphErrors(len(SFarray_alleta), ptarray, SFarray_alleta, pterrarray, SFerrarray_alleta)
    histogram["Eff_jet_pt"] = TGraphErrors(len(effarray_alleta), ptarray, effarray_alleta, pterrarray, efferrarray_alleta)
    histogram["MC_Eff_jet_pt"] = TGraphErrors(len(MCeffarray_alleta), ptarray, MCeffarray_alleta, pterrarray, MCefferrarray_alleta)
    
    cv1 = TCanvas("SF_jet_pt","SF_jet_pt",700,700)
    histogram["SF_jet_pt"].Draw("ap")
    histogram["SF_jet_pt"].SetName("SF_jet_pt")
    histogram["SF_jet_pt"].GetHistogram().SetXTitle("Jet p_{T} [GeV/c]")
    histogram["SF_jet_pt"].GetHistogram().SetYTitle("SF_{b}=#epsilon^{"+OP+"}_{data}/#epsilon^{"+OP+"}_{MC}")
    
    cv2 = TCanvas("Eff_jet_pt","Eff_jet_pt",700,700)
    histogram["Eff_jet_pt"].Draw("ap")
    histogram["Eff_jet_pt"].SetName("Eff_jet_pt")
    histogram["Eff_jet_pt"].GetHistogram().SetXTitle("Jet p_{T} [GeV/c]")
    histogram["Eff_jet_pt"].GetHistogram().SetYTitle("b-tag efficiency #epsilon^{"+OP+"}_{data}")

    cv3 = TCanvas("MC_Eff_jet_pt","MC_Eff_jet_pt",700,700)
    histogram["MC_Eff_jet_pt"].Draw("ap")
    histogram["MC_Eff_jet_pt"].SetName("MC_Eff_jet_pt")
    histogram["MC_Eff_jet_pt"].GetHistogram().SetXTitle("Jet p_{T} [GeV/c]")
    histogram["MC_Eff_jet_pt"].GetHistogram().SetYTitle("b-tag efficiency #epsilon^{"+OP+"}_{MC}")

                    
    outfilename = "plotPerformanceDB.root"
    outputroot = TFile( outfilename, "RECREATE")
        
    for key in histogram.keys():
        histogram[key].Write()

    outputroot.Close()
                    
    
    raw_input ("Enter to quit:")
     

if __name__ == '__main__':
    main()
            
