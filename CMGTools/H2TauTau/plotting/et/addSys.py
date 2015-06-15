import ROOT
import os

def main():
    inputName = 'eleTau.root'
    outputName = 'eleTauSys.root'
    
    inFile = ROOT.TFile(inputName)
    outFile = ROOT.TFile(outputName, 'recreate')

    keeper = []
    for dirKey in inFile.GetListOfKeys():

        subDir = inFile.GetDirectory(dirKey.GetName())
        for histKey in subDir.GetListOfKeys():
            # outFile.cd()
            # print histKey

            ### --- QCD UNCERTAINTY --- ###
            if 'QCD' == histKey.GetName() and ('1jet' in dirKey.GetName() or 'vbf' in dirKey.GetName()):
                inFile.cd()
                obj = subDir.Get(histKey.GetName())
                print 'Adding systematic uncertainty for category', dirKey.GetName(), 'histogram',  histKey.GetName()

                objUp = obj.Clone()
                objDown = obj.Clone()
                keeper.append(objUp)
                keeper.append(objDown)

                for iBin in range(1, objUp.GetNbinsX()):

                    if objUp.GetBinCenter(iBin) < 50.:
                        print objUp.GetBinContent(iBin)
                        print objUp.GetBinCenter(iBin)
                        objUp.SetBinContent(iBin, objUp.GetBinContent(iBin) * 1.1)
                        objDown.SetBinContent(iBin, objDown.GetBinContent(iBin) * 0.9)
                # QCDShape_etau_1jet_medium_8TeV
                outFile.cd()
                subdir = outFile.mkdir(dirKey.GetName())
                subdir.cd()
                objUp.Write(histKey.GetName() + '_CMS_htt_QCDShape_' + dirKey.GetName().replace('eleTau', 'etau') + '_8TeVUp')
                objDown.Write(histKey.GetName() + '_CMS_htt_QCDShape_' + dirKey.GetName().replace('eleTau', 'etau') + '_8TeVDown')
                # CMS_htt_QCDShape_

            ### --- Uncorrelated tau ES uncertainty --- ###
            if 'CMS_scale_t_etau_8TeV' in histKey.GetName():
                inFile.cd()
                obj = subDir.Get(histKey.GetName())
                print 'Adding tau ES uncorrelated systematic uncertainty for category', dirKey.GetName(), 'histogram',  histKey.GetName()

                objUncorr = obj.Clone()
                keeper.append(objUncorr)

                # QCDShape_etau_1jet_medium_8TeV
                outFile.cd()
                subdir = outFile.GetDirectory(dirKey.GetName())
                if not subdir:
                    subdir = outFile.mkdir(dirKey.GetName())
                subdir.cd()
                cat = '_low'
                if 'jet_medium' in dirKey.GetName():
                    cat = '_medium'
                elif 'high' in dirKey.GetName():
                    cat = '_high'
                objUncorr.Write(histKey.GetName().replace('CMS_scale_t_etau_8TeV', 'CMS_scale_t_etau{cat}_8TeV'.format(cat=cat)))

    outFile.Write()
    outFile.Close()
    os.system('hadd -f htt_et.inputs-sm-8TeV.root eleTau.root eleTauSys.root')
    return inFile
    

if __name__ == '__main__':
    main()
