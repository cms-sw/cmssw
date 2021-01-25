from .adapt_to_new_backend import *
dqmitems={}

def hltBTVlayout(i, p, *rows): i['HLT/Layouts/BTV/' + p] = rows


triggers =  [
            'BTagMu_DiJet/BTagMu_AK4DiJet20_Mu5',
            'BTagMu_DiJet/BTagMu_AK4DiJet40_Mu5',
            'BTagMu_DiJet/BTagMu_AK4DiJet70_Mu5',
            'BTagMu_DiJet/BTagMu_AK4DiJet110_Mu5',
            'BTagMu_DiJet/BTagMu_AK4DiJet170_Mu5',
            'BTagMu_DiJet/BTagMu_AK8DiJet170_Mu5',

            'BTagMu_Jet/BTagMu_AK4Jet300_Mu5',
            'BTagMu_Jet/BTagMu_AK8Jet300_Mu5',
            ]


for trigger in triggers:
    hltBTVlayout(dqmitems, trigger.split('/')[1],
        [{'path': 'HLT/BTV/{}/effic_bjetPt_1_variableBinning'.format(trigger),
            'description': 'efficiency vs bjet Pt',
            'draw': { 'withref': 'no', 'xmax': '700' }},
        {'path': 'HLT/BTV/{}/effic_bjetEtaPhi_1'.format(trigger),
            'description': 'efficiency vs bjet Eta/Phi',
            'draw': { 'withref': 'no', 'zmin': '0', 'zmax': '1', 'drawopts': 'colz'}},
        {'path': 'HLT/BTV/{}/effic_bjetEta_1_variableBinning'.format(trigger),
            'description': 'efficiency vs bjet multiplicity',
            'draw': { 'withref': 'no', 'ymin': '0', 'ymax': '0.4'}},
        {'path': 'HLT/BTV/{}/effic_bjetMulti'.format(trigger),
            'description': 'efficiency vs bjet multiplicity',
            'draw': { 'withref': 'no' }},
        {'path': 'HLT/BTV/{}/effic_bjetCSV_1'.format(trigger),
            'description': 'efficiency vs bjet1 CSV',
            'draw': { 'withref': 'no' }}
        ],
        [{'path': 'HLT/BTV/{}/effic_muPt_1_variableBinning'.format(trigger),
            'description': 'efficiency vs muon Pt',
            'draw': { 'withref': 'no', 'xmax': '400'}},
        {'path': 'HLT/BTV/{}/effic_muEtaPhi_1'.format(trigger),
            'description': 'efficiency vs muon Eta/Phi',
            'draw': { 'withref': 'no', 'zmin': '0', 'zmax': '1', 'drawopts': 'colz'}},
        {'path': 'HLT/BTV/{}/effic_muMulti'.format(trigger),
            'description': 'efficiency vs muon multiplicity',
            'draw': { 'withref': 'no' }},
        {'path': 'HLT/BTV/{}/effic_DeltaR_jet_Mu'.format(trigger),
            'description': 'efficiency vs DeltaR(jet, muon)',
            'draw': { 'withref': 'no' }}
        ])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
