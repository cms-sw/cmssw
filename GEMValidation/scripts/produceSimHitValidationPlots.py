import sys

from ROOT import *

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)


if __name__ == "__main__":  

  inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/tempDir/CMSSW_6_2_0_pre7/src/GEMCode/GEMValidation/test/gem_sh_ana.test.root'
  inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/testForMarcello/test/CMSSW_6_2_0_SLHC5/src/gem_sh_ana.root'
  targetDir = './'
  
  ## extension for figures - add more?
  ext = ".png"
  
  ## GEM system settings
  nregion = 2
  nlayer = 2
  npart = 8
  
  ## Trees
  analyzer = "GEMSimHitAnalyzer"
  simHits = "GEMSimHits"
  simHits = "ME0SimHits"
  simTracks = "Tracks"

  ## muon selection
  muonSelection = [TCut("TMath::Abs(particleType)==13"),TCut("TMath::Abs(particleType)!=13"),TCut("TMath::Abs(particleType)==13 || TMath::Abs(particleType)!=13")]
  titlePrefix = ["Muon","Non muon","All"]
  histSuffix = ["_muon","_nonmuon","_all"]

  ## Style
  gStyle.SetStatStyle(0);

  ## input
  file = TFile.Open(inputFile)
  if not file:
    sys.exit('Input ROOT file %s is missing.' %(inputFile))

  dirAna = file.Get(analyzer)
  if not dirAna:
    sys.exit('Directory %s does not exist.' %(dirAna))
    
  treeHits = dirAna.Get(simHits)
  if not treeHits:
    sys.exit('Tree %s does not exist.' %(treeHits))
  
  for i in range(len(muonSelection)):

    sel = muonSelection[i]
    pre = titlePrefix[i]
    suff = histSuffix[i]

    ####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####
    if simHits=="ME0SimHits":
      
      draw_occ(targetDir, "sh_xy_rm1_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rm1,l1,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rm1_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rm1,l2,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rm1_l3" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer3;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rm1,l3,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rm1_l4" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer4;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rm1,l4,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rm1_l5" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer5;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rm1,l5,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rm1_l6" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer6;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rm1,l6,sel), "COLZ")
      
      draw_occ(targetDir, "sh_xy_rp1_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rp1,l1,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rp1_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]", 
 	     "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rp1,l2,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rp1_l3" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer3;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rp1,l3,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rp1_l4" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer4;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rp1,l4,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rp1_l5" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer5;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rp1,l5,sel), "COLZ")
      draw_occ(targetDir, "sh_xy_rp1_l6" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer6;globalX [cm];globalY [cm]", 
               "h_", "(120,-280,280,120,-280,280)", "globalY:globalX", AND(rp1,l6,sel), "COLZ")
      
      draw_occ(targetDir, "sh_zr_rm1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1;globalZ [cm];globalR [cm]", 
               "h_", "(80,-555,-515,120,20,280)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rm1,sel), "COLZ")
      draw_occ(targetDir, "sh_zr_rp1" + suff, ext, treeHits, pre + " SimHit occupancy: region1;globalZ [cm];globalR [cm]", 
               "h_", "(80,515,555,120,20,280)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rp1,sel), "COLZ")

      draw_1D(targetDir, "sh_tof_rm1_l1" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer1;Time of flight [ns];entries", 
              "h_", "(40,15,19)", "timeOfFlight", AND(rm1,l1,sel))
      draw_1D(targetDir, "sh_tof_rm1_l2" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer2;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l2,sel))
      draw_1D(targetDir, "sh_tof_rm1_l3" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer3;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l3,sel))
      draw_1D(targetDir, "sh_tof_rm1_l4" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer4;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l4,sel))
      draw_1D(targetDir, "sh_tof_rm1_l5" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer5;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l5,sel))
      draw_1D(targetDir, "sh_tof_rm1_l6" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer6;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rm1,l6,sel))
      
      draw_1D(targetDir, "sh_tof_rp1_l1" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer1;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l1,sel))
      draw_1D(targetDir, "sh_tof_rp1_l2" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer2;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l2,sel))
      draw_1D(targetDir, "sh_tof_rp1_l3" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer3;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l3,sel))
      draw_1D(targetDir, "sh_tof_rp1_l4" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer4;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l4,sel))
      draw_1D(targetDir, "sh_tof_rp1_l5" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer5;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l5,sel))
      draw_1D(targetDir, "sh_tof_rp1_l6" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer6;Time of flight [ns];entries", 
              "h_", "(40,15,22)", "timeOfFlight", AND(rp1,l6,sel))

    ####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####ME0####

    draw_occ(targetDir, "sh_xy_rm1_st1_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, station1, layer1;globalX [cm];globalY [cm]",
             "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l1,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rm1_st1_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, station1, layer2;globalX [cm];globalY [cm]",
             "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rm1,st1,l2,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_st1_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region1, station1 ,layer1;globalX [cm];globalY [cm]",
             "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l1,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_st1_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region1, station1, layer2;globalX [cm];globalY [cm]",
             "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", AND(rp1,st1,l2,sel), "COLZ")
    
    draw_occ(targetDir, "sh_xy_rm1_st2_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, station2, layer1;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st2,l1,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rm1_st2_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, station2, layer2;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st2,l2,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_st2_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region1, station2, layer1;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st2,l1,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_st2_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region1, station2, layer2;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st2,l2,sel), "COLZ")
   
    draw_occ(targetDir, "sh_xy_rm1_st3_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, station3, layer1;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st3,l1,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rm1_st3_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, station3, layer2;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rm1,st3,l2,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_st3_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region1, station3, layer1;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st3,l1,sel), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_st3_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region1, station3, layer2;globalX [cm];globalY [cm]",
             "h_", "(100,-280,280,100,-280,280)", "globalY:globalX", AND(rp1,st3,l2,sel), "COLZ")

    draw_occ(targetDir, "sh_zr_rm1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1;globalZ [cm];globalR [cm]",
             "h_", "(200,-573,-564,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rm1,sel), "COLZ")
    draw_occ(targetDir, "sh_zr_rp1" + suff, ext, treeHits, pre + " SimHit occupancy: region1;globalZ [cm];globalR [cm]",
             "h_", "(200,564,573,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", AND(rp1,sel), "COLZ")

    draw_1D(targetDir, "sh_tof_rm1_l1" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer1;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", AND(rm1,l1,sel))
    draw_1D(targetDir, "sh_tof_rm1_l2" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer2;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", AND(rm1,l2,sel))
    draw_1D(targetDir, "sh_tof_rp1_l1" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer1;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", AND(rp1,l1,sel))
    draw_1D(targetDir, "sh_tof_rp1_l2" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer2;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", AND(rp1,l2,sel))

    ## momentum plot
    c = TCanvas("c","c",600,600)
    c.Clear()
    treeHits.Draw("pabs>>hh(200,0.,200.)",sel)
    h = TH1F(gDirectory.Get("hh"))
    gPad.SetLogx(0)
    gPad.SetLogy(1)
    h.SetTitle(pre + " SimHits absolute momentum;Momentum [GeV/c];entries")       
    h.SetLineWidth(2)
    h.SetLineColor(kBlue)
    h.Draw("")        
    c.SaveAs(targetDir +"sh_momentum" + suff + ext)
    

    ## PDGID
    draw_1D(targetDir, "sh_pdgid" + suff, ext, treeHits, pre + " SimHit PDG Id;PDG Id;entries", 
   	    "h_", "(200,-100.,100.)", "particleType", sel)


    ## eta occupancy plot
    h = TH1F("h", pre + " SimHit occupancy in eta partitions; occupancy in #eta partition; entries",4*npart,1.,1.+4*npart)
    entries = treeHits.GetEntriesFast()
    for jentry in xrange(entries):
      ientry = treeHits.LoadTree( jentry )
      if ientry < 0:
        break
      nb = treeHits.GetEntry( jentry )
      if nb <= 0:
        continue
      if treeHits.layer==2:
        layer = npart
      else:
        layer = 0
      if treeHits.region==1:
        region = 2.*npart
      else:
        region = 0
      if i==0:
        if abs(treeHits.particleType)==13:
          h.Fill(treeHits.roll + layer + region)
      elif i==1:
        if not abs(treeHits.particleType)!=13:
          h.Fill(treeHits.roll + layer + region)
      elif i==2:
          h.Fill(treeHits.roll + layer + region)
      
    c = TCanvas("c","c",600,600)
    c.Clear()  
    gPad.SetLogx(0)
    gPad.SetLogy(0)
    ibin = 1
    for iregion in range(1,3):
      if iregion ==1:
        region = "-"
      else:
        region = "+"
      for ilayer in range(1,3):
        for ipart in range(1,npart+1):
          h.GetXaxis().SetBinLabel(ibin,"%s%d%d"% (region,ilayer,ipart))
          ibin = ibin + 1
    h.SetMinimum(0.)
    h.SetLineWidth(2)
    h.SetLineColor(kBlue)
    h.Draw("")        
    c.SaveAs(targetDir +"sh_globalEta" + suff + ext)
    
    ## energy loss plot
    h = TH1F("h","",60,0.,6000.)
    entries = treeHits.GetEntriesFast()
    for jentry in xrange(entries):
      ientry = treeHits.LoadTree( jentry )
      if ientry < 0:
        break
      nb = treeHits.GetEntry( jentry )
      if nb <= 0:
        continue
      if i==0:
        if abs(treeHits.particleType)==13:
          h.Fill( treeHits.energyLoss*1.e9 )
      elif i==1:
        if not abs(treeHits.particleType)!=13:
          h.Fill( treeHits.energyLoss*1.e9 )
      elif i==2:
        h.Fill( treeHits.energyLoss*1.e9 )
        
##   gStyle.SetStatStyle(0)
##   gStyle.SetOptStat(1110)
##   gPad.SetLogx(0)
##   gPad.SetLogy(0)
  c = TCanvas("c","c",600,600)
  c.Clear()  
  h.SetTitle(pre + " SimHit energy loss;Energy loss [eV];entries")
  gPad.SetLogx(0)
  gPad.SetLogy(0)
  h.SetMinimum(0.)
  h.SetLineWidth(2)
  h.SetLineColor(kBlue)
  h.Draw("")        
  c.SaveAs(targetDir + "sh_energyloss" + suff + ext)

  treeTracks = dirAna.Get(simTracks)
  if not treeTracks:
    sys.exit('Tree %s does not exist.' %(treeTracks))

  draw_geff(targetDir, "eff_eta_track_sh_gem_l1or2", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), OR(ok_gL1sh,ok_gL2sh), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_sh_gem_l1", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_sh_gem_l2", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_sh_gem_l1and2", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), AND(ok_gL1sh,ok_gL2sh), "P", kBlue)
    
  draw_geff(targetDir, "eff_phi_track_sh_gem_l1or2", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, OR(ok_gL1sh,ok_gL2sh), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_sh_gem_l1", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, ok_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_sh_gem_l2", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, ok_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_sh_gem_l1and2", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, AND(ok_gL1sh,ok_gL2sh), "P", kBlue)


  
  draw_geff(targetDir, "eff_lx_track_sh_gem_l1_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), ok_trk_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_sh_gem_l2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), ok_trk_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_sh_gem_l1or2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_sh_gem_l1and2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_even", TCut(""), AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)

  draw_geff(targetDir, "eff_ly_track_sh_gem_l1_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, ok_trk_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_sh_gem_l2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, ok_trk_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_sh_gem_l1or2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_sh_gem_l1and2_even", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_even", ok_lx_even, AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)

  draw_geff(targetDir, "eff_lx_track_sh_gem_l1_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), ok_trk_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_sh_gem_l2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), ok_trk_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_sh_gem_l1or2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(targetDir, "eff_lx_track_sh_gem_l1and2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localX [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_lx_odd", TCut(""), AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)

  draw_geff(targetDir, "eff_ly_track_sh_gem_l1_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, ok_trk_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_sh_gem_l2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, ok_trk_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_sh_gem_l1or2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, OR(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)
  draw_geff(targetDir, "eff_ly_track_sh_gem_l1and2_odd", ext, treeTracks,
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack localy [cm];Eff.", 
            "h_", "(100,-100,100)", "gem_ly_odd", ok_lx_odd, AND(ok_trk_gL1sh,ok_trk_gL2sh), "P", kBlue)


    
  draw_1D(targetDir, "track_pt", ext, treeTracks, "Track p_{T};Track p_{T} [GeV];Entries", "h_", "(100,0,200)", "pt", "")
  draw_1D(targetDir, "track_eta", ext, treeTracks, "Track |#eta|;Track |#eta|;Entries", "h_", "(100,1.5,2.2)", "eta", "")
  draw_1D(targetDir, "track_phi", ext, treeTracks, "Track #phi;Track #phi [rad];Entries", "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", "")

  
