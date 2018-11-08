import ROOT
from DataFormats.FWLite import Handle, Events

events_c = Events('output_test_DDX.root')

handleJ  = Handle ("std::vector<pat::Jet>")
#labelJ = ("slimmedJetsAK8","","PAT")
#labelJ = ("updatedPatJets","","PATtest")
labelJ = ("selectedUpdatedPatJets","","PATtest")
#labelJ = ("updatedPatJetsTransientCorrected","","PATtest")


#    Type                                  Module                      Label            Process     
#    -----------------------------------------------------------------------------------------------
#    edm::ValueMap<float>                  "offlineSlimmedPrimaryVertices"   ""               "PAT"       
#    vector<pat::Jet>                      "slimmedJetsAK8"            ""               "PAT"       
#    vector<pat::Jet>                      "slimmedJetsAK8PFPuppiSoftDropPacked"   "SubJets"        "PAT"       
#    vector<reco::Vertex>                  "offlineSlimmedPrimaryVertices"   ""               "PAT"       
#    vector<reco::VertexCompositePtrCandidate>    "slimmedSecondaryVertices"   ""               "PAT"       
#    edm::OwnVector<reco::BaseTagInfo,edm::ClonePolicy<reco::BaseTagInfo> >    "selectedUpdatedPatJets"    "tagInfos"       "PATtest"   
#    edm::OwnVector<reco::BaseTagInfo,edm::ClonePolicy<reco::BaseTagInfo> >    "updatedPatJets"            "tagInfos"       "PATtest"   
#    edm::OwnVector<reco::BaseTagInfo,edm::ClonePolicy<reco::BaseTagInfo> >    "updatedPatJetsTransientCorrected"   "tagInfos"       "PATtest"   
#    vector<CaloTower>                     "selectedUpdatedPatJets"    "caloTowers"     "PATtest"   
#    vector<pat::Jet>                      "selectedUpdatedPatJets"    ""               "PATtest"   
#    vector<pat::Jet>                      "updatedPatJets"            ""               "PATtest"   
#    vector<pat::Jet>                      "updatedPatJetsTransientCorrected"   ""               "PATtest"   
#    vector<reco::BoostedDoubleSVTagInfo>    "pfBoostedDoubleSVAK8TagInfos"   ""               "PATtest"   
#    vector<reco::FeaturesTagInfo<btagbtvdeep::DeepDoubleBFeatures> >    "pfDeepDoubleBTagInfos"     ""               "PATtest"   
#    vector<reco::GenJet>                  "selectedUpdatedPatJets"    "genJets"        "PATtest"   
#    vector<reco::PFCandidate>             "selectedUpdatedPatJets"    "pfCandidates"   "PATtest"   

h_probQ_ddb = ROOT.TH1F('h_probQ_ddb', ';prob Q;', 40, 0., 1.)
h_probH_ddb = ROOT.TH1F('h_probH_ddb', ';prob H;', 40, 0., 1.)
h_probQplusH_ddb = ROOT.TH1F('h_probQplusH_ddb', ';prob Q + H;', 44, 0., 1.1)

h_probQ_ddc = ROOT.TH1F('h_probQ_ddc', ';prob Q;', 40, 0., 1.)
h_probH_ddc = ROOT.TH1F('h_probH_ddc', ';prob H;', 40, 0., 1.)
h_probQplusH_ddc = ROOT.TH1F('h_probQplusH_ddc', ';prob Q + H;', 44, 0., 1.1)

for iev,event in enumerate(events_c):
    event.getByLabel (labelJ, handleJ)
    jets = handleJ.product()
    for jet in jets  :
	if jet.pt() < 300 or jet.pt() > 2000: continue
	if jet.mass() < 40 or jet.mass() > 200: continue

        print jet.pt(), jet.mass()
        print "DDB", jet.bDiscriminator("pfDeepDoubleBvLJetTags:probQCD"), jet.bDiscriminator("pfDeepDoubleBvLJetTags:probHbb") , jet.bDiscriminator("pfDeepDoubleBvLJetTags:probHcc")
        print "DDCvL", jet.bDiscriminator("pfDeepDoubleCvLJetTags:probQCD"), jet.bDiscriminator("pfDeepDoubleCvLJetTags:probHcc") , jet.bDiscriminator("pfDeepDoubleCvLJetTags:probHbb")
        print "DDCvL", jet.bDiscriminator("pfMassIndependentDeepDoubleCvLJetTags:probQCD"), jet.bDiscriminator("pfMassIndependentDeepDoubleCvLJetTags:probHcc") , jet.bDiscriminator("pfDeepDoubleMassIndependentCvLJetTags:probHbb")
        print "DDCvB", jet.bDiscriminator("pfDeepDoubleCvBJetTags:probHbb"), jet.bDiscriminator("pfDeepDoubleCvBJetTags:probHcc") , jet.bDiscriminator("pfDeepDoubleCvBJetTags:probQCD")
        print "DDCvB", jet.bDiscriminator("pfMassIndependentDeepDoubleCvBJetTags:probHbb"), jet.bDiscriminator("pfMassIndependentDeepDoubleCvBJetTags:probHcc") , jet.bDiscriminator("pfMassIndependentDeepDoubleCvBJetTags:probQCD")
        h_probQ_ddb.Fill(jet.bDiscriminator("pfDeepDoubleBvLJetTags:probQCD"))
        h_probH_ddb.Fill(jet.bDiscriminator("pfDeepDoubleBvLJetTags:probHbb"))
        h_probQ_ddc.Fill(jet.bDiscriminator("pfDeepDoubleCvLJetTags:probQCD"))
        h_probH_ddc.Fill(jet.bDiscriminator("pfDeepDoubleCvLJetTags:probHcc"))
    if iev > 10000: break

c1a = ROOT.TCanvas()
h_probH_ddb.Draw("HISTO")
h_probH_ddb.SetLineColor(632)
h_probH_ddb.SetLineStyle(10)
h_probQ_ddb.Draw("SAME")
c1a.Draw()
c1a.SaveAs("ProbQ_vc_vb.png")

c1b = ROOT.TCanvas()
h_probH_ddc.Draw("HISTO")
h_probH_ddc.SetLineColor(632)
h_probH_ddc.SetLineStyle(10)
h_probQ_ddc.Draw("SAME")
c1b.Draw()
c1b.SaveAs("ProbH_vc_vb.png")
