import os
from CMGTools.H2TauTau.proto.plotter.datacards import *


for mssm in [True, False]:

    prefixes = ['']
    if mssm:
        prefixes = ['2013', 'default', 'fine']

    for prefix in prefixes:
        if not mssm:

            inFiles = [
            'eleTau_0jet_high.root',
            'eleTau_0jet_medium.root',
            'eleTau_1jet_high_mediumhiggs.root',
            # 'eleTau_0jet_low.root',
            # 'eleTau_1jet_high_lowhiggs.root',
            'eleTau_1jet_medium.root',
            'eleTau_vbf_loose.root',
            'eleTau_vbf_tight.root',
            ]

            systs = ['_CMS_scale_t_etau_8TeV', '_QCDscale_ggH1in', '_CMS_htt_ZLScale_etau_8TeV', '_CMS_htt_WShape_etau_0jet_medium_8TeV', '_CMS_htt_WShape_etau_0jet_high_8TeV', '_CMS_htt_WShape_etau_1jet_medium_8TeV', '_CMS_htt_WShape_etau_1jet_high_mediumhiggs_8TeV', '_CMS_htt_WShape_etau_vbf_loose_8TeV', '_CMS_htt_WShape_etau_vbf_tight_8TeV',
            '_CMS_htt_ZLScale_etau_8TeVSmeared']
        else:
            inFiles = [
            '{prefix}_eleTau_btag.root'.format(prefix=prefix),
            '{prefix}_eleTau_nobtag.root'.format(prefix=prefix),
            ]

            systs = ['_CMS_scale_t_etau_8TeV', '_QCDscale_ggH1in', '_CMS_htt_ZLScale_etau_8TeV']



        for f in inFiles:
            target = f.replace('.root', '.tmp.root')
            inFilesSys = [f]
            inFilesSys += [f.replace('.root', sys+'Up.root') for sys in systs if os.path.isfile(f.replace('.root', sys+'Up.root'))]
            inFilesSys += [f.replace('.root', sys+'Down.root') for sys in systs if os.path.isfile(f.replace('.root', sys+'Down.root'))]
            os.system('hadd {target} '.format(target=target)+' '.join(inFilesSys))

        inFiles = [f.replace('.root', '.tmp.root') for f in inFiles]

        merge(inFiles, prefix=prefix)

        if mssm:
            os.system('mv eleTau.root {prefix}.root'.format(prefix=prefix))

        for f in inFiles:
            os.remove(f)

    if mssm:
        os.system('hadd -f htt_et.inputs-mssm-8TeV-0.root default.root fine.root')
        os.system('hadd -f htt_et.inputs-mssm-8TeV-0-fb.root 2013.root fine.root')
