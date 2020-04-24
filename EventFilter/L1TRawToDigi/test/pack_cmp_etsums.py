import ROOT
import sys

from DataFormats.FWLite import Events, Handle

ROOT.gROOT.SetBatch()

def compare_bx_vector(xs, ys):
    x_total_size = xs.getLastBX() - xs.getFirstBX() + 1
    y_total_size = ys.getLastBX() - ys.getLastBX() + 1

    if x_total_size != y_total_size:
        print "> BX count mismatch:", x_total_size, "vs", y_total_size
        print ">", xs.getFirstBX(), ",", ys.getFirstBX()
        print ">", xs.getLastBX(), ",", ys.getLastBX()
        return

    for bx in range(xs.getFirstBX(), xs.getLastBX() + 1):
        x_size = xs.size(bx)
        y_size = ys.size(bx)

        if x_size != y_size:
            print ">> BX size mismatch:", x_size, "vs", y_size, "@", bx

        for i in range(x_size):
            x = xs.at(bx, i)
            for m in range(y_size):
                y = ys.at(bx, m)
                
                if x.getType() == y.getType():
                    print ">>> Pt :", x.hwPt(), "vs", y.hwPt()
                    print ">>> eta :", x.hwPhi(), "vs", y.hwPhi()
                    print ">>> overflow :", (x.hwQual()&0x1), "vs", (y.hwQual()&0x1)
                    print ">>> type :", x.getType(), "vs", y.getType()
                    
                    if x.hwPt() != y.hwPt(): print ">>> Pt mismatch", x.hwPt(), "vs", y.hwPt()
                    if x.hwPhi() != y.hwPhi(): print ">>> Phi mismatch", x.hwPhi(), "vs", y.hwPhi()
                    if (x.hwQual()&0x1) != (y.hwQual()&0x1): print ">>> Qual mismatch",(x.hwQual()&0x1), "vs", (y.hwQual()&0x1)
                    if x.getType() != y.getType(): print ">>> Type mismatch", x.hwPt(), "vs", y.hwPt()
                    yield x,y
                    
                

events = Events(sys.argv[1])

spares_in = Handle('BXVector<l1t::CaloSpare>')
spares_out = Handle('BXVector<l1t::CaloSpare>')

egammas_in = Handle('BXVector<l1t::EGamma>')
egammas_out = Handle('BXVector<l1t::EGamma>')

etsums_in = Handle('BXVector<l1t::EtSum>')
etsums_out = Handle('BXVector<l1t::EtSum>')

jets_in = Handle('BXVector<l1t::Jet>')
jets_out = Handle('BXVector<l1t::Jet>')

taus_in = Handle('BXVector<l1t::Tau>')
taus_out = Handle('BXVector<l1t::Tau>')

# in_label = "Layer2Phys"
in_label = ("caloStage1FinalDigis", "")
tau_label = ("caloStage1FinalDigis", "isoTaus")
out_label = "l1tRawToDigi"

in_ring_label = ("caloStage1FinalDigis", "HFRingSums")
out_ring_label = ("l1tRawToDigi", "HFRingSums")

in_bit_label = ("caloStage1FinalDigis", "HFBitCounts")
out_bit_label = ("l1tRawToDigi", "HFBitCounts")

for event in events:
    print "< New event"
    event.getByLabel(in_ring_label, spares_in)
    event.getByLabel(in_label, egammas_in)
    event.getByLabel(in_label, etsums_in)
    event.getByLabel(in_label, jets_in)
    event.getByLabel(tau_label, taus_in)

    event.getByLabel(out_ring_label, spares_out)
    event.getByLabel(out_label, egammas_out)
    event.getByLabel(out_label, etsums_out)
    event.getByLabel(out_label, jets_out)
    event.getByLabel(out_label, taus_out)

    print "Checking etsums"
    for a, b in compare_bx_vector(etsums_in.product(), etsums_out.product()):
        if a.getType() != b.getType():
            print ">>> Type different:", a.getType(), "vs", b.getType()
