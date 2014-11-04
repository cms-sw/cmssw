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

        for i in range(min(x_size, y_size)):
            x = xs.at(bx, i)
            y = ys.at(bx, i)

            if x.hwPt() != y.hwPt():
                print ">>> Pt mismatch:", x.hwPt(), "vs", y.hwPt()
            if x.hwEta() != y.hwEta():
                print ">>> Eta mismatch:", x.hwEta(), "vs", y.hwEta()
            if x.hwPhi() != y.hwPhi():
                print ">>> Phi mismatch:", x.hwPhi(), "vs", y.hwPhi()
            if x.hwQual() != y.hwQual():
                print ">>> Qual mismatch:", x.hwQual(), "vs", y.hwQual()
            if x.hwIso() != y.hwIso():
                print ">>> Iso mismatch:", x.hwIso(), "vs", y.hwIso()

            yield x, y

        for j in range(min(x_size, 0), min(x_size, y_size)):
            x = xs.at(bx, j)
            y = ys.at(bx, j)
            print ">>>> ({0} @ {1}, {2} : {3} - {4}) vs ({5} @ {6}, {7} : {8} - {9})".format(
                    x.hwPt(), x.hwEta(), x.hwPhi(), x.hwQual(), x.hwIso(),
                    y.hwPt(), y.hwEta(), y.hwPhi(), y.hwQual(), y.hwIso())

        print "<< Compared", x_size, "quantities"

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

    print "Checking egammas"
    for a, b in compare_bx_vector(egammas_in.product(), egammas_out.product()):
        pass

    print "Checking spare ring"
    for a, b in compare_bx_vector(spares_in.product(), spares_out.product()):
        if a.getType() != b.getType():
            print ">>> Type different:", a.getType(), "vs", b.getType()

    event.getByLabel(in_bit_label, spares_in)
    event.getByLabel(out_bit_label, spares_out)

    print "Checking spare bits"
    for a, b in compare_bx_vector(spares_in.product(), spares_out.product()):
        if a.getType() != b.getType():
            print ">>> Type different:", a.getType(), "vs", b.getType()

    print "Checking etsums"
    for a, b in compare_bx_vector(etsums_in.product(), etsums_out.product()):
        if a.getType() != b.getType():
            print ">>> Type different:", a.getType(), "vs", b.getType()

    print "Checking jets"
    for a, b in compare_bx_vector(jets_in.product(), jets_out.product()):
        pass

    print "Checking taus"
    for a, b in compare_bx_vector(taus_in.product(), taus_out.product()):
        pass
