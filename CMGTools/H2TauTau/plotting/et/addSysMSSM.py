import ROOT
import os

def main():
    for mode in ['', '-fb']:
        inputName = 'htt_et.inputs-mssm-8TeV-0{mode}.root'.format(mode=mode)
        outputName = 'htt_et.inputs-mssm-8TeV-0{mode}_sys.root'.format(mode=mode)
        
        inFile = ROOT.TFile(inputName)
        outFile = ROOT.TFile(outputName, 'recreate')

        keeper = []
        for dirKey in inFile.GetListOfKeys():

            subDir = inFile.GetDirectory(dirKey.GetName())
            for histKey in subDir.GetListOfKeys():
                # outFile.cd()
                # print histKey

                ### --- QCD UNCERTAINTY --- ###
                if histKey.GetName().startswith('QCD'):
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
                    outFile.cd()
                    subdir = outFile.GetDirectory(dirKey.GetName())
                    if not subdir: subdir = outFile.mkdir(dirKey.GetName())
                    subdir.cd()
                    outName = 'QCD_CMS_htt_QCDShape_' + dirKey.GetName().replace('eleTau', 'etau')
                    fb = ''
                    if 'fine_binning' in histKey.GetName(): fb = '_fine_binning'
                    objUp.Write(outName + '_8TeVUp' + fb)
                    objDown.Write(outName+ '_8TeVDown' + fb)

        outFile.Write()
        outFile.Close()
        os.system('cp htt_et.inputs-mssm-8TeV-0{mode}.root htt_et.inputs-mssm-8TeV-0{mode}_tmp.root'.format(mode=mode))
        os.system('hadd -f htt_et.inputs-mssm-8TeV-0{mode}.root htt_et.inputs-mssm-8TeV-0{mode}_tmp.root htt_et.inputs-mssm-8TeV-0{mode}_sys.root'.format(mode=mode))
  

if __name__ == '__main__':
    main()
