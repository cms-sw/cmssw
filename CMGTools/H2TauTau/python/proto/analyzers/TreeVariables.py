from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi

from CMGTools.H2TauTau.proto.analyzers.tauIDs import tauIDs

class Variable():
    def __init__(self, name, function=None, type=float):
        self.name = name
        self.function = function
        if function is None:
            # Note: works for attributes, not member functions
            self.function = lambda x : getattr(x, self.name, -999.) 
        self.type = float

# event variables
event_vars = [
    Variable('run', type=int),
    Variable('lumi', type=int),
    Variable('event', lambda ev : ev.eventId, type=int),
    Variable('pass_leptons', lambda ev : ev.isSignal, type=int),
    Variable('veto_dilepton', lambda ev : ev.leptonAccept, type=int),
    Variable('veto_thirdlepton', lambda ev : ev.thirdLeptonVeto, type=int),
    Variable('n_jets', lambda ev : len(ev.cleanJets30), type=int),
    Variable('n_jets_20', lambda ev : len(ev.cleanJets), type=int),
    Variable('n_bjets', lambda ev : len(ev.cleanBJets), type=int),
    Variable('n_jets_csvl', lambda ev : sum(1 for jet in ev.cleanJets if jet.btagWP('CSVv2IVFL')), type=int),
    Variable('n_vertices', lambda ev : len(ev.vertices), type=int),
    Variable('rho', lambda ev : ev.rho),
    Variable('weight', lambda ev : ev.eventWeight),
    Variable('weight_vertex', lambda ev : ev.vertexWeight),
    Variable('weight_embed', lambda ev : ev.embedWeight),
    Variable('weight_njet', lambda ev : ev.NJetWeight),
    Variable('weight_hqt', lambda ev : ev.higgsPtWeight),
    Variable('weight_hqt_up', lambda ev : ev.higgsPtWeightUp),
    Variable('weight_hqt_down', lambda ev : ev.higgsPtWeightDown),
]

# di-tau object variables
ditau_vars = [
    Variable('mvis', lambda dil : dil.mass()),
    Variable('svfit_mass', lambda dil : dil.svfitMass()),
    Variable('svfit_mass_error', lambda dil : dil.svfitMassError()),
    Variable('svfit_pt', lambda dil : dil.svfitPt()),
    Variable('svfit_pt_error', lambda dil : dil.svfitPtError()),
    Variable('svfit_eta', lambda dil : dil.svfitEta()),
    Variable('svfit_phi', lambda dil : dil.svfitPhi()),
    Variable('svfit_met_pt', lambda dil : dil.svfitMET().Rho() if hasattr(dil, 'svfitMET') else -999.),
    Variable('svfit_met_e', lambda dil : dil.svfitMET().mag2() if hasattr(dil, 'svfitMET') else -999.),
    Variable('svfit_met_phi', lambda dil : dil.svfitMET().phi() if hasattr(dil, 'svfitMET') else -999.),
    Variable('svfit_met_eta', lambda dil : dil.svfitMET().eta() if hasattr(dil, 'svfitMET') else -999.),
    Variable('pzeta_met', lambda dil : dil.pZetaMET()),
    Variable('pzeta_vis', lambda dil : dil.pZetaVis()),
    Variable('pzeta_disc', lambda dil : dil.pZetaDisc()),
    Variable('mt', lambda dil : dil.mTLeg2()),
    Variable('mt_leg2', lambda dil : dil.mTLeg2()),
    Variable('mt_leg1', lambda dil : dil.mTLeg1()),
    Variable('met_cov00', lambda dil : dil.mvaMetSig(0, 0)),
    Variable('met_cov01', lambda dil : dil.mvaMetSig(0, 1)),
    Variable('met_cov10', lambda dil : dil.mvaMetSig(1, 0)),
    Variable('met_cov11', lambda dil : dil.mvaMetSig(1, 1)),
    Variable('met_phi', lambda dil : dil.met().phi()),
    Variable('met_px', lambda dil : dil.met().px()),
    Variable('met_py', lambda dil : dil.met().py()),
    Variable('met_pt', lambda dil : dil.met().pt()),
    Variable('pthiggs', lambda dil : (dil.leg1().p4() + dil.leg2().p4() + dil.met().p4()).pt()),
    Variable('delta_phi_l1_l2', lambda dil : deltaPhi(dil.leg1().phi(), dil.leg2().phi())),
    Variable('delta_eta_l1_l2', lambda dil : abs(dil.leg1().eta() - dil.leg2().eta())),
    Variable('delta_r_l1_l2', lambda dil : deltaR(dil.leg1().eta(), dil.leg1().phi(), dil.leg2().eta(), dil.leg2().phi())),
    Variable('delta_phi_l1_met', lambda dil : deltaPhi(dil.leg1().phi(), dil.met().phi())),
    Variable('delta_phi_l2_met', lambda dil : deltaPhi(dil.leg2().phi(), dil.met().phi())),
]

# generic particle
particle_vars = [
    Variable('pt', lambda p: p.pt()),
    Variable('eta', lambda p: p.eta()),
    Variable('phi', lambda p: p.phi()),
    Variable('charge', lambda p: p.charge() if hasattr(p, 'charge') else 0), # charge may be non-integer for gen particles
    Variable('mass', lambda p: p.mass()),
]

# generic lepton
lepton_vars = [
    Variable('reliso05', lambda lep : lep.relIso(dBetaFactor=0.5, allCharged=0)),
    Variable('dxy', lambda lep : lep.dxy()),
    Variable('dz', lambda lep : lep.dz()),
    Variable('weight'),
    Variable('weight_trigger', lambda lep : getattr(lep, 'triggerWeight', -999.)),
    Variable('eff_trigger_data', lambda lep : getattr(lep, 'triggerEffData', -999.)),
    Variable('eff_trigger_mc', lambda lep : getattr(lep, 'triggerEffMC', -999.)),
    Variable('weight_rec_eff', lambda lep : getattr(lep, 'recEffWeight', -999.)),
]

# electron
electron_vars = [
    Variable('eid_nontrigmva_loose', lambda ele : ele.mvaIDRun2("NonTrigPhys14", "Loose")),
    Variable('eid_nontrigmva_tight', lambda ele : ele.mvaIDRun2("NonTrigPhys14", "Tight")),
    Variable('eid_veto', lambda ele : ele.cutBasedId('POG_PHYS14_25ns_v1_Veto')),
    Variable('eid_loose', lambda ele : ele.cutBasedId('POG_PHYS14_25ns_v1_Loose')),
    Variable('eid_medium', lambda ele : ele.cutBasedId('POG_PHYS14_25ns_v1_Medium')),
    Variable('eid_tight', lambda ele : ele.cutBasedId('POG_PHYS14_25ns_v1_Tight')),
    Variable('nhits_missing', lambda ele : ele.physObj.gsfTrack().hitPattern().numberOfHits(1), int),
    Variable('pass_conv_veto', lambda ele : ele.passConversionVeto()),
]

# muon
muon_vars = [
    Variable('muonid_loose', lambda muon : muon.muonID('POG_ID_Loose')),
    Variable('muonid_medium', lambda muon : muon.muonID('POG_ID_Medium')),
    Variable('muonid_tight', lambda muon : muon.muonID('POG_ID_Tight')),
    Variable('muonid_tightnovtx', lambda muon : muon.muonID('POG_ID_TightNoVtx')),
    Variable('muonid_highpt', lambda muon : muon.muonID('POG_ID_HighPt')),
]

# tau
tau_vars = [
    Variable('decayMode', lambda tau : tau.decayMode()),
    Variable('zImpact', lambda tau : tau.zImpact())
]
for tau_id in tauIDs:
    if type(tau_id) is str:
        # Need to use eval since functions are otherwise bound to local
        # variables
        tau_vars.append(Variable(tau_id, eval('lambda tau : tau.tauID("{id}")'.format(id=tau_id))))
    else:
        sum_id_str = ' + '.join('tau.tauID("{id}")'.format(id=tau_id[0].format(wp=wp)) for wp in tau_id[1])
        tau_vars.append(Variable(tau_id[0].format(wp=''), 
            eval('lambda tau : ' + sum_id_str), int))


# jet
jet_vars = [
    # JAN - only one PU mva working point, but we may want to add more
    # run in our skimming step
    # (for which Jet.py would have to be touched again)
    Variable('mva_pu', lambda jet : jet.puMva('pileupJetIdFull:full53xDiscriminant')),
    Variable('id_loose', lambda jet : jet.looseJetId()),
    Variable('id_pu', lambda jet : jet.puJetId()),
    Variable('mva_btag', lambda jet : jet.btagMVA),
    Variable('area', lambda jet : jet.jetArea()),
    Variable('flavour_parton', lambda jet : jet.partonFlavour()),
    Variable('csv', lambda jet : jet.btag('combinedInclusiveSecondaryVertexV2BJetTags')),
    Variable('rawfactor', lambda jet : jet.rawFactor()),
    Variable('genjet_pt', lambda jet : jet.matchedGenJet.pt() if hasattr(jet, 'matchedGenJet') and jet.matchedGenJet else -999.),
]

# gen info
geninfo_vars = [
    Variable('geninfo_nup', lambda ev : ev.NUP if hasattr(ev, 'NUP') else -1, type=int),
    Variable('geninfo_tt', type=int),
    Variable('geninfo_mt', type=int),
    Variable('geninfo_et', type=int),
    Variable('geninfo_ee', type=int),
    Variable('geninfo_mm', type=int),
    Variable('geninfo_em', type=int),
    Variable('geninfo_EE', type=int),
    Variable('geninfo_MM', type=int),
    Variable('geninfo_TT', type=int),
    Variable('geninfo_LL', type=int),
    Variable('geninfo_fakeid', type=int),
    Variable('geninfo_has_w', type=int),
    Variable('geninfo_has_z', type=int),
    Variable('geninfo_mass'),
    Variable('genmet_pt'),
    Variable('genmet_eta'),
    Variable('genmet_e'),
    Variable('genmet_px'),
    Variable('genmet_py'),
    Variable('genmet_phi'),
]

vbf_vars = [
    Variable('mjj'),
    Variable('deta'),
    Variable('n_central', lambda vbf : len(vbf.centralJets), int),
    Variable('jdphi', lambda vbf : vbf.dphi),
    Variable('dijetpt'),
    Variable('dijetphi'),
    Variable('dphidijethiggs'),
    Variable('mindetajetvis', lambda vbf : vbf.visjeteta),
]
