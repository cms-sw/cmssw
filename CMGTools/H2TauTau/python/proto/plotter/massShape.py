from ROOT import TH1F

keeper = []

hRef = None
hNum = 1

def massShape(tree, mvacut, nbins=15, min=0, max=300):
    global hNum, hRef
    
    hname = 'h_{mvacut}'.format(mvacut=mvacut)
    h = TH1F(hname, ';m_{sv} (GeV)', nbins, min, max)
    tree.Project(hname, 'svfitMass', cat_VBF.replace('0.5', str(mvacut))), 
    h.Sumw2()
    h.Scale(1/h.Integral())
    h.SetLineWidth(2)
    h.SetLineColor(hNum)
    keeper.append(h)
    if hRef is None:
        hRef = h
        hRef.Draw()
    else:
        opt = 'same'
        if hNum == 1:
            opt = ''
        h.Divide(hRef)
        h.Draw()
        hNum += 1
    gPad.SaveAs('massShape_{cut}.png'.format(cut=str(mvacut).replace('-','m')))
        

