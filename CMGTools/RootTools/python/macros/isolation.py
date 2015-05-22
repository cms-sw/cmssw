# this macro reads pat muons.

# mus = 'patMuons_selectedPatMuonsAK5LC__PAT'
# mus = 'patMuons_selectedPatMuonsAK5__PAT'

countAlias = False
if countAlias:
    countAlias = {
        'Nvertices':'int_vertexSize__',
        'Nmus':'int_muonSelSize__'
        }
    aliases = AliasSetter(events, countAlias, 'COUNT')

def setAliases( tree, mus): 
    tree.SetAlias('ch',mus + '.obj[0].chargedHadronIso()')
    tree.SetAlias('nh',mus + '.obj[0].neutralHadronIso()')
    tree.SetAlias('ph',mus + '.obj[0].photonIso()')
    
    tree.SetAlias('chpu',mus + '.obj[0].puChargedHadronIso()')
    
    tree.SetAlias('comb', 'ch+nh+ph')
    
    tree.SetAlias('pt', mus + '.obj[0].pt()')


class ProbeCut:
    def __init__( self, leg = 'leg1', isoCut = 0.15, dBetaFactor = 0.5):
        self.leg = leg
        self.isoCut = isoCut
        self.dBetaFactor = dBetaFactor
    def __str__( self ):
        return 'diMus.obj[0].%s().relIso(%f)<%f' % (self.leg, self.dBetaFactor, self.isoCut)

class TagCut:
    def __init__( self, leg = 'leg2', mZmin = 86, mZmax = 96, ptMin = 20, etaMax=2.4,
                  isoMax = 0.1, muId = 'cuts_vbtfmuon', nvMin=0, nvMax=999):
        self.leg = leg
        self.mZmin = mZmin
        self.mZmax = mZmax
        self.ptMin = ptMin
        self.etaMax = etaMax
        self.isoMax = isoMax
        self.muId = muId
        self.nvMin = nvMin
        self.nvMax = nvMax
    def __str__( self ):
        evCut = 'Nvertices.obj>=%5.2f && Nvertices.obj<=%5.2f' % (self.nvMin, self.nvMax) 
        zCut =  '   diMus.@obj.size()==1'
        zCut += '&& diMus.obj[0].mass()>%f && diMus.obj[0].mass()<%f' % (self.mZmin, self.mZmax) 
        zCut += '&& diMus.obj[0].leg1().charge()*diMus.obj[0].leg2().charge()<0'
        tagCutLeg1 = 'diMus.obj[0].%s().pt()>%f' % (self.leg, self.ptMin) 
        tagCutLeg1 += '&& abs(diMus.obj[0].%s().eta())<%f' % (self.leg, self.etaMax) 
        tagCutLeg1 += '&& diMus.obj[0].%s().relIso()<%f' % (self.leg, self.isoMax)
        tagCutLeg1 += '&& diMus.obj[0].%s().getSelection("%s")' % (self.leg, self.muId)
        
        # applying some of the cuts also on the probe leg
        probeLeg = 'leg1'
        if self.leg == 'leg1':
            probeLeg = 'leg2'
        tagCutLeg2 = 'diMus.obj[0].%s().getSelection("%s")' % (probeLeg, self.muId)
        tagCutLeg2 += '&& diMus.obj[0].%s().pt()>%f' % (probeLeg, self.ptMin) 
        tagCutLeg2 += '&& abs(diMus.obj[0].%s().eta())<%f' % (probeLeg, self.etaMax) 
        
        tagCut = evCut + '&&' + zCut + ' && ' + tagCutLeg1 + ' && ' + tagCutLeg2
        return tagCut


from CMGTools.RootTools.TagAndProbe import *
from CMGTools.RootTools.Style import *

# some initialization


setAliases( events, 'mus')
nEvents = 9999999

# wolfgang
isoCut = 0.15
setEventList( events, 'Nmus.obj>=2' )


# ---- no dbeta correction

# Nv

probeCut = ProbeCut('leg1', dBetaFactor=0.0)
tagCut = TagCut('leg2')

h1NoDBeta = TagAndProbeBothLegs('h1NoDBeta')
h1NoDBeta.fillHistos( events, 'Nvertices.obj', 20, 0, 20, probeCut, tagCut, nEvents)
h1NoDBeta.formatHistos( sBlack, ';N_{vertices}')
h1NoDBeta.leg1.formatHistos( sRed )
h1NoDBeta.leg2.formatHistos( sBlue )
h1NoDBeta.write()

# eta 

hEtaNoDBeta = TagAndProbeBothLegs('hEtaNoDBeta')
hEtaNoDBeta.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaNoDBeta.formatHistos( sBlack, ';#eta')
hEtaNoDBeta.write()

# eta, NV 0-4

tagCut.nvMin = 0
tagCut.nvMax = 4
hEtaNV0_4 = TagAndProbeBothLegs('hEtaNV0_4')
hEtaNV0_4.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaNV0_4.formatHistos( sBlack, ';#eta')
hEtaNV0_4.write()

# eta, NV 5_8

tagCut.nvMin = 5
tagCut.nvMax = 8
hEtaNV5_8 = TagAndProbeBothLegs('hEtaNV5_8')
hEtaNV5_8.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaNV5_8.formatHistos( sRed, ';#eta')
hEtaNV5_8.write()

# eta, NV 9_11

tagCut.nvMin = 9
tagCut.nvMax = 11
hEtaNV9_11 = TagAndProbeBothLegs('hEtaNV9_11')
hEtaNV9_11.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaNV9_11.formatHistos( sBlue, ';#eta')
hEtaNV9_11.write()

# eta, NV 12-

tagCut.nvMin = 12
tagCut.nvMax = 999
hEtaNV12_inf = TagAndProbeBothLegs('hEtaNV12_inf')
hEtaNV12_inf.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaNV12_inf.formatHistos( sGreen, ';#eta')
hEtaNV12_inf.write()


# ---- dbeta correction

# Nv

probeCut = ProbeCut('leg1', dBetaFactor=0.5)
tagCut = TagCut('leg2')

h1DBeta = TagAndProbeBothLegs('h1DBeta')
h1DBeta.fillHistos( events, 'Nvertices.obj', 20, 0, 20, probeCut, tagCut, nEvents)
h1DBeta.formatHistos( sBlue, ';N_{vertices}')
h1DBeta.write()

# eta 

hEtaDBeta = TagAndProbeBothLegs('hEtaDBeta')
hEtaDBeta.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaDBeta.formatHistos( sBlue, ';#eta')
hEtaDBeta.write()


# eta, NV 0-4

tagCut.nvMin = 0
tagCut.nvMax = 4
hEtaDBetaNV0_4 = TagAndProbeBothLegs('hEtaDBetaNV0_4')
hEtaDBetaNV0_4.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaDBetaNV0_4.formatHistos( sBlack, ';#eta')
hEtaDBetaNV0_4.write()

# eta, NV 5_8

tagCut.nvMin = 5
tagCut.nvMax = 8
hEtaDBetaNV5_8 = TagAndProbeBothLegs('hEtaDBetaNV5_8')
hEtaDBetaNV5_8.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaDBetaNV5_8.formatHistos( sRed, ';#eta')
hEtaDBetaNV5_8.write()

# eta, NV 9_11

tagCut.nvMin = 9
tagCut.nvMax = 11
hEtaDBetaNV9_11 = TagAndProbeBothLegs('hEtaDBetaNV9_11')
hEtaDBetaNV9_11.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaDBetaNV9_11.formatHistos( sBlue, ';#eta')
hEtaDBetaNV9_11.write()

# eta, NV 12-

tagCut.nvMin = 12
tagCut.nvMax = 999
hEtaDBetaNV12_inf = TagAndProbeBothLegs('hEtaDBetaNV12_inf')
hEtaDBetaNV12_inf.fillHistos( events, 'diMus.obj[0].leg1().eta()', 10, -2.5, 2.5, probeCut, tagCut, nEvents)
hEtaDBetaNV12_inf.formatHistos( sGreen, ';#eta')
hEtaDBetaNV12_inf.write()



# effect of the choice of the leg on the eff vs nvert =====================================

# if probe is leg2, tag is leg1, which has a higher pT and thus a lower relative isolation
# so there are more selected events on which the probe efficiency can be evaluated.
# the errors are larger cause they are binomial
# leg2 efficiency is lower because it has a lower pT, hence a higher relative isolation

cLeg = TCanvas('cLeg','Effect of leg choice', 750,700)
formatPad( cLeg )

h1NoDBeta.leg1.hEff.GetYaxis().SetRangeUser(0.8,1.05)
h1NoDBeta.leg1.hEff.Draw()
h1NoDBeta.leg2.hEff.Draw('same')
h1NoDBeta.sum.hEff.Draw('same')

x=0.1
legend1 = TLegend(0.18+x,0.16+x,0.43+x,0.33+x)
legend1.AddEntry(h1NoDBeta.leg1.hEff, 'leg 1')
legend1.AddEntry(h1NoDBeta.leg2.hEff, 'leg 2')
legend1.AddEntry(h1NoDBeta.sum.hEff, 'both legs')
legend1.Draw('same')

# effect of the dbeta corrections =========================================================

# Nv

cDbeta = TCanvas('cDbeta','dbeta effect', 750,700)
formatPad( cDbeta )
cDbeta.SetGridy()
cDbeta.SetGridx()

h1NoDBeta.sum.hEff.Draw()
h1DBeta.sum.hEff.Draw('same')

x=0.1
legend2 = TLegend(0.18+x,0.16+x,0.53+x,0.33+x)
h1NoDBeta.sum.hEff.GetYaxis().SetRangeUser(0.8,1.05)
legend2.AddEntry(h1NoDBeta.sum.hEff, 'no dbeta cor')
legend2.AddEntry(h1DBeta.sum.hEff, 'dbeta cor')
legend2.Draw('same')

cDbeta.SaveAs('eff_Nvert.png')

# eta

cDbetaEta = TCanvas('cDbetaEta','dbeta effect, vs eta', 750,700)
formatPad( cDbetaEta )
cDbetaEta.SetGridy()
cDbetaEta.SetGridx()

hEtaNoDBeta.sum.hEff.GetYaxis().SetRangeUser(0.8,1.05)
hEtaNoDBeta.sum.hEff.Draw()
hEtaDBeta.sum.hEff.Draw('same')

x=0.1
legend3 = TLegend(0.18+x,0.16+x,0.53+x,0.33+x)
h1NoDBeta.sum.hEff.GetYaxis().SetRangeUser(0.8,1.05)
legend3.AddEntry(hEtaNoDBeta.sum.hEff, 'no dbeta cor')
legend3.AddEntry(hEtaDBeta.sum.hEff, 'dbeta cor')
legend3.Draw('same')

cDbetaEta.SaveAs('eff_eta.png')


# efficiency vs eta for different ranges in NVertex. no dbeta corrections ===================

cEtaNv = TCanvas('cEtaNv','eff vs eta for ranges in NVertex', 750,700)
formatPad( cEtaNv )
hEtaNV0_4.sum.hEff.GetYaxis().SetRangeUser(0.8,1.05)
hEtaNV0_4.sum.hEff.Draw()
hEtaNV5_8.sum.hEff.Draw("same")
hEtaNV9_11.sum.hEff.Draw("same")
hEtaNV12_inf.sum.hEff.Draw("same")

x=0.1
legend4 = TLegend(0.18+x,0.16+x,0.53+x,0.33+x)
h1NoDBeta.sum.hEff.GetYaxis().SetRangeUser(0.8,1.05)
legend4.AddEntry(hEtaNV0_4.sum.hEff, '0 - 4 vertices')
legend4.AddEntry(hEtaNV5_8.sum.hEff, '5 - 8 vertices')
legend4.AddEntry(hEtaNV9_11.sum.hEff, '9 - 11 vertices')
legend4.AddEntry(hEtaNV12_inf.sum.hEff, '12 - #infty vertices')
legend4.Draw('same')

cEtaNv.SaveAs('eff_eta_vertexRanges.png')


# efficiency vs eta for different ranges in NVertex. dbeta corrections ===================

cEtaDBetaNv = TCanvas('cEtaDBetaNv','eff vs eta for ranges in NVertex', 750,700)
formatPad( cEtaDBetaNv )
hEtaDBetaNV0_4.sum.hEff.GetYaxis().SetRangeUser(0.8,1.05)
hEtaDBetaNV0_4.sum.hEff.Draw()
hEtaDBetaNV5_8.sum.hEff.Draw("same")
hEtaDBetaNV9_11.sum.hEff.Draw("same")
hEtaDBetaNV12_inf.sum.hEff.Draw("same")

x=0.1
legend4 = TLegend(0.18+x,0.16+x,0.53+x,0.33+x, '#Delta#beta corrections ON')
h1NoDBeta.sum.hEff.GetYaxis().SetRangeUser(0.8,1.05)
legend4.AddEntry(hEtaDBetaNV0_4.sum.hEff, '0 - 4 vertices')
legend4.AddEntry(hEtaDBetaNV5_8.sum.hEff, '5 - 8 vertices')
legend4.AddEntry(hEtaDBetaNV9_11.sum.hEff, '9 - 11 vertices')
legend4.AddEntry(hEtaDBetaNV12_inf.sum.hEff, '12 - #infty vertices')
legend4.Draw('same')

cEtaDBetaNv.SaveAs('eff_eta_vertexRanges_dbeta.png')



# pt of the first and second leg

cLegPt = TCanvas('cLegPt','pt distribution for legs 1 and 2', 750,700)
formatPad( cLegPt )
events.Draw('diMus.obj[0].leg1().pt()>>h_pt1', 'diMus.obj[0].mass()>70', 'goff', 2000)
h_pt1 = events.GetHistogram()
sBlack.formatHisto( h_pt1, ';p_{T} (GeV)')
events.Draw('diMus.obj[0].leg2().pt()>>h_pt2', 'diMus.obj[0].mass()>70', 'goff', 2000)
h_pt2 = events.GetHistogram()
sBlue.formatHisto( h_pt2, '')
h_pt1.SetStats(0)
h_pt2.SetStats(0)
h_pt1.Draw()
h_pt2.Draw('same')
x=0.4
legend2 = TLegend(0.18+x,0.16+x,0.43+x,0.33+x)
legend2.AddEntry(h_pt1, 'leg 1')
legend2.AddEntry(h_pt2, 'leg 2')
legend2.Draw('same')


# relIso of the first and second leg

cLegIso = TCanvas('cLegIso','relIso distribution for legs 1 and 2', 750,700)
formatPad( cLegIso )
gPad.SetLogy()
h_relIso1 = TH1F('h_relIso1', '', 100,0,1)
h_relIso2 = TH1F('h_relIso2', '', 100,0,1)
events.Draw('diMus.obj[0].leg1().relIso()>>h_relIso1', 'diMus.obj[0].mass()>70', 'goff', 10000)
sBlack.formatHisto( h_relIso1, ';rel iso')
events.Draw('diMus.obj[0].leg2().relIso()>>h_relIso2', 'diMus.obj[0].mass()>70', 'goff', 10000)
sBlue.formatHisto( h_relIso2, '')
h_relIso1.SetStats(0)
h_relIso2.SetStats(0)
h_relIso1.Draw()
h_relIso2.Draw('same')
x=0.4
legend2 = TLegend(0.18+x,0.16+x,0.43+x,0.33+x)
legend2.AddEntry(h_relIso1, 'leg 1')
legend2.AddEntry(h_relIso2, 'leg 2')
legend2.Draw('same')


# Z mass spectrum for tagged events

cMass = TCanvas('cMass','Z mass for tagged events', 750,700)
formatPad( cMass )
h_mass1 = TH1F('h_mass1', '', 100,40,140)
h_mass2 = TH1F('h_mass2', '', 100,40,140)
events.Draw('diMus.obj[0].mass()>>h_mass1', h1.leg2.tagCut, 'goff', 10000)
sBlack.formatHisto( h_mass1, ';m_{Z} (GeV)')
h_mass1.Draw()
