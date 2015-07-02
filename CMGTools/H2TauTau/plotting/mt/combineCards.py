import os
from CMGTools.H2TauTau.proto.plotter.datacards import *


for mssm in [True, False]:

    prefixes = ['']
    if mssm:
        prefixes = ['2013', 'default', 'fine']

    for prefix in prefixes:
        if not mssm:

            inFiles = [
            'muTau_0jet_high.root',
            'muTau_0jet_medium.root',
            'muTau_1jet_high_mediumhiggs.root',
            'muTau_0jet_low.root',
            'muTau_1jet_high_lowhiggs.root',
            'muTau_1jet_medium.root',
            'muTau_vbf_loose.root',
            'muTau_vbf_tight.root',
            ]

            systs = ['_CMS_scale_t_mutau_8TeV', '_QCDscale_ggH1in', '_CMS_htt_ZLScale_mutau_8TeV']
        else:
            inFiles = [
            '{prefix}_muTau_btag.root'.format(prefix=prefix),
            '{prefix}_muTau_nobtag.root'.format(prefix=prefix),
            ]

            systs = ['_CMS_scale_t_mutau_8TeV', '_QCDscale_ggH1in', '_CMS_htt_ZLScale_mutau_8TeV']



        for f in inFiles:
            target = f.replace('.root', '.tmp.root')
            inFilesSys = [f]
            inFilesSys += [f.replace('.root', sys+'Up.root') for sys in systs]
            inFilesSys += [f.replace('.root', sys+'Down.root') for sys in systs]
            os.system('hadd {target} '.format(target=target)+' '.join(inFilesSys))

        inFiles = [f.replace('.root', '.tmp.root') for f in inFiles]

        merge(inFiles, prefix=prefix)

        if mssm:
            os.system('mv muTau.root {prefix}.root'.format(prefix=prefix))

        for f in inFiles:
            os.remove(f)

    if mssm:
        os.system('hadd -f htt_mt.inputs-mssm-8TeV-0.root default.root fine.root')
        os.system('hadd -f htt_mt.inputs-mssm-8TeV-0-fb.root 2013.root fine.root')
