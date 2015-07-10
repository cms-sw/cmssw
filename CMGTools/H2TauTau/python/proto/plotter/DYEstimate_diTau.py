import imp
import inspect
from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC_diTau import *
from CMGTools.RootTools.RootInit import *
import math

antiLeptons  = ' & l2againstElectronNewLooseMVA3>0.5 '
antiLeptons += ' & l1LooseEle>0.5  & l2LooseEle>0.5  '
antiLeptons += ' & l1LooseMu >0.5  & l2LooseMu >0.5  '

def embeddedScaleFactor(anaDir, selCompsNoSignal, weightsNoSignal, selCompsDataMass, weightsDataMass, weight, prong, trigMatch=True, susy=False):
    
    cut    = ' l1Pt>45 & l2Pt>45 & abs(l1Eta)<2.1 & abs(l2Eta)<2.1 & diTauCharge==0'
    cut   += ' & l1RawDB3HIso<1.0 & l2RawDB3HIso<1.0'
    cut   += antiLeptons
    cut   += ' & muon1Pt == -1 & electron1Pt == -1 '
    #cut   += ' & nbJets == 0 '

    cutDY  = cut 
    cutEmb = cut
    
    cutDY  += '& isZtt'
    cutEmb += '& isZtt'
    if trigMatch and susy:
      cutDY += ' & (l1TrigMatched_diTau > 0.5 & l2TrigMatched_diTau > 0.5)'
    if trigMatch and not susy:
      cutDY += ' & ((l1TrigMatched_diTau > 0.5  & l2TrigMatched_diTau   >0.5) || \
                    (l1TrigMatched_diTauJet>0.5 & l2TrigMatched_diTauJet>0.5  &  jetTrigMatched_diTauJet>0.5 & jet1Pt>50 & abs(jet1Eta)<3.0) )'

    if prong :
      cutDY  += ' & l1DecayMode<3  & l2DecayMode<3'
      cutEmb += ' & l1DecayMode<3  & l2DecayMode<3'
      cutDY.replace( 'abs(jet1Eta)<3.0 & jet1Pt>50.','')
      cutEmb.replace('abs(jet1Eta)<3.0 & jet1Pt>50.','')

    ## if diTau is matched, apply diTau data/MC correction factor 
    weight  = '   ( ((l1TrigMatched_diTau>0.5) && (l2TrigMatched_diTau>0.5))*triggerWeight_diTau +\
                    ((l1TrigMatched_diTau<0.5) || (l2TrigMatched_diTau<0.5))                     ) '
    ## if diTauJet is matched, apply diTau data/MC correction factor 
    weight += ' * ( ((l1TrigMatched_diTauJet>0.5) && (l2TrigMatched_diTauJet>0.5) && (jetTrigMatched_diTauJet)>0.5)*triggerWeight_diTauJet +\
                    ((l1TrigMatched_diTauJet<0.5) || (l2TrigMatched_diTauJet<0.5) || (jetTrigMatched_diTauJet)<0.5)                        ) '
    ## apply the diTau data turn on on embed sample
    weight += ' * ( (embedWeight == 1) + (embedWeight != 1)*triggerEffData_diTau )'
    ## stitching weight
    weight += ' * ( weight * (1/(triggerWeight_diTau*triggerWeight_diTauJet)) )'
    ## scale down real taus with decay mode 1 prong no pi zero by 0.88
    weight += ' * ( ((isRealTau==1) * ((l1DecayMode==0)*0.88 + (l1DecayMode!=0))) + (isRealTau==0) ) ' 
    weight += ' * ( ((isRealTau==1) * ((l2DecayMode==0)*0.88 + (l2DecayMode!=0))) + (isRealTau==0) ) ' 

    inclusiveForEmbeddedNormalizationDY    = H2TauTauDataMC('svfitMass', 
                                                            anaDir, 
                                                            selCompsNoSignal, 
                                                            weightsNoSignal ,
     			                                    35,0,350,
     			                                    cut    = cutDY  , 
     			                                    weight = weight ,
     			                                    embed  = False  ,
     			                                    susy   = susy )
    
    inclusiveForEmbeddedNormalizationEmbed = H2TauTauDataMC('svfitMass', 
                                                            anaDir, 
                                                            selCompsNoSignal, 
                                                            weightsNoSignal ,
     			                                    35,0,350        ,
     			                                    cut    = cutEmb , 
     			                                    weight = weight ,
     			                                    embed  = False  ,
     			                                    susy   = susy )
    
    ### sum up all the embedded components
    embeddedHist = None
    for name,comp in selCompsNoSignal.items():
        if comp.isEmbed:
	     if embeddedHist == None:
	         embeddedHist = copy.deepcopy(inclusiveForEmbeddedNormalizationEmbed.Hist(name))
             else:
	         embeddedHist.Add(inclusiveForEmbeddedNormalizationEmbed.Hist(name))
   
    print "lumi                         ", inclusiveForEmbeddedNormalizationDY.intLumi
    print "DY events in inclusive       ", inclusiveForEmbeddedNormalizationDY.Hist("DYJets").weighted.Integral(0,35)
    print "DY entries in inclusive      ", inclusiveForEmbeddedNormalizationDY.Hist("DYJets").weighted.GetEntries()
    print "Embedded events in inclusive ", embeddedHist.weighted.Integral(0,35)
    print "Embedded entries in inclusive", embeddedHist.weighted.GetEntries()
    
    embeddedScaleFactor = inclusiveForEmbeddedNormalizationDY.Hist("DYJets").weighted.Integral(0,35)/embeddedHist.weighted.Integral(0,35)
    
    print "embeddedScaleFactor", embeddedScaleFactor

    # plotting
    embeddedHist.Scale(embeddedScaleFactor)

    ymax = max(inclusiveForEmbeddedNormalizationDY.Hist("DYJets").GetMaximum(),embeddedHist.GetMaximum())
    inclusiveForEmbeddedNormalizationDY.Hist("DYJets").weighted.Draw("HISTe")
    inclusiveForEmbeddedNormalizationDY.Hist("DYJets").weighted.GetYaxis().SetRangeUser(0,ymax*1.5)
    embeddedHist.weighted.Draw("HISTeSAME")

    gPad.SaveAs("inclusiveForEmbeddedNormalization.png")
    gPad.WaitPrimitive()

    for name,comp in selCompsNoSignal.items():
        if comp.isEmbed:
	     comp.embedFactor = embeddedScaleFactor
    for mass,comps in selCompsDataMass.items():
      for name,comp in comps.items():
        if comp.isEmbed:
	     comp.embedFactor = embeddedScaleFactor

def zeeScaleFactor(anaDir, selCompsNoSignal, weightsNoSignal, selCompsDataMass, weightsDataMass, weight, embed, susy=False):
    # Data/MC scale factors for e->tau fake rate from 2012 ICHEP Object approval presentation: 0.85 for Barrel, 0.65 for Endcap
    #inclusiveForEmbeddedNormalizationZeeBB = H2TauTauDataMC('svfitMass', anaDir, selCompsNoSignal, weightsNoSignal,
    inclusiveForEmbeddedNormalizationZeeBB = H2TauTauDataMC('visMass', anaDir, selCompsNoSignal, weightsNoSignal,
     			    #30,70,160,
     			    50,70,120,
     			    cut    = 'isZee & abs(l1Eta)<1.5 & abs(l2Eta)<1.5 & l1Pt>45 & l2Pt>45 & abs(l1Eta)<2.1 & abs(l2Eta)<2.1 & diTauCharge==0 & l1RawDB3HIso<1.0 & l2RawDB3HIso<1.0 & l1againstElectronNewLooseMVA3<0.5 & l2againstElectronNewLooseMVA3<0.5', 
     			    weight = weight,
     			    embed  = embed,
     			    susy   = susy)
    print "Data events in boosted ee"  , inclusiveForEmbeddedNormalizationZeeBB.Hist("Data").weighted.Integral()
    print "DYJets events in boosted ee", (inclusiveForEmbeddedNormalizationZeeBB.Hist("DYJets").weighted.Integral()+inclusiveForEmbeddedNormalizationZeeBB.Hist("DYJets_ZL").weighted.Integral())

    zeeScaleFactor = inclusiveForEmbeddedNormalizationZeeBB.Hist("Data").weighted.Integral()/ \
        (inclusiveForEmbeddedNormalizationZeeBB.Hist("DYJets").weighted.Integral()+inclusiveForEmbeddedNormalizationZeeBB.Hist("DYJets_ZL").weighted.Integral())
    
    #zeeScaleFactor = 1.
    
    print "zeeScaleFactor", zeeScaleFactor, "+-", math.sqrt(pow(0.2,2)+pow(0.2,2))*zeeScaleFactor

    ymax = max(inclusiveForEmbeddedNormalizationZeeBB.Hist("Data").GetMaximum(),(inclusiveForEmbeddedNormalizationZeeBB.Hist("DYJets").GetMaximum()+inclusiveForEmbeddedNormalizationZeeBB.Hist("DYJets_ZL").GetMaximum()))*1.5
    inclusiveForEmbeddedNormalizationZeeBB.DrawStack("HIST",0,300,0,ymax)
    
    gPad.SaveAs("inclusiveForZeeNormalization.png")
    gPad.WaitPrimitive()

