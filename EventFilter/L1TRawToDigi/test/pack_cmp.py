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
            #if ((x.hwQual()>>0)&0x1) != ((y.hwQual()>>0)&0x1):
            #    print ">>> Qual bit 0 mismatch:", ((x.hwQual()>>0)&0x1), "vs", ((y.hwQual()>>0)&0x1)
            if ((x.hwQual()>>1)&0x1) != ((y.hwQual()>>1)&0x1):
                print ">>> Qual bit 1 mismatch:", ((x.hwQual()>>1)&0x1), "vs", ((y.hwQual()>>1)&0x1)
            if x.hwIso() != y.hwIso():
                print ">>> Iso mismatch:", x.hwIso(), "vs", y.hwIso()

            yield x, y

        # for j in range(min(x_size, 0), min(x_size, y_size)):
        #     x = xs.at(bx, j)
        #     y = ys.at(bx, j)
        #     print ">>>> ({0} @ {1}, {2} : {3}, {4} - {5}) vs ({6} @ {7}, {8} : {9}, {10} - {11})".format(
        #             x.hwPt(), x.hwEta(), x.hwPhi(), ((x.hwQual()>>0)&0x1), ((x.hwQual()>>1)&0x1), x.hwIso(),
        #             y.hwPt(), y.hwEta(), y.hwPhi(), ((y.hwQual()>>0)&0x1), ((y.hwQual()>>1)&0x1), y.hwIso())

        print "<< Compared", x_size, "quantities"

class Test(object):
    def __init__(self, msg, type, inlabel, outlabel, tests):
        self.msg = msg
        self.inhandle = Handle(type)
        self.outhandle = Handle(type)
        self.inlabel = inlabel
        self.outlabel = outlabel,
        self.tests = tests or []

    def __call__(self, event):
        event.getByLabel(*(list(self.inlabel) + [self.inhandle]))
        event.getByLabel(*(list(self.outlabel) + [self.outhandle]))

        print self.msg
        for a, b in compare_bx_vector(self.inhandle.product(), self.outhandle.product()):
            for t in self.tests:
                t(a, b)

def test_type(a, b):
    if a.getType() != b.getType():
        print ">>> Type different:", a.getType(), "vs", b.getType()

events = Events(sys.argv[1])

run = [
        Test(
            'Checking spare rings',
            'BXVector<l1t::CaloSpare>',
            ('caloStage1FinalDigis', 'HFRingSums'),
            ('l1tRawToDigi', 'HFRingSums'),
            [test_type]
        ),
        Test(
            'Checking spare bits',
            'BXVector<l1t::CaloSpare>',
            ('caloStage1FinalDigis', 'HFBitCounts'),
            ('l1tRawToDigi', 'HFBitCounts'),
            [test_type]
        ),
        Test(
            'Checking EG',
            'BXVector<l1t::EGamma>',
            ('caloStage1FinalDigis',),
            ('l1tRawToDigi',),
            []
        ),
        Test(
            'Checking EtSum',
            'BXVector<l1t::EtSum>',
            ('caloStage1FinalDigis',),
            ('l1tRawToDigi',),
            []
        ),
        Test(
            'Checking Jets',
            'BXVector<l1t::Jet>',
            ('caloStage1FinalDigis',),
            ('l1tRawToDigi',),
            []
        ),
        Test(
            'Checking Taus',
            'BXVector<l1t::Tau>',
            ('caloStage1FinalDigis', 'rlxTaus'),
            ('l1tRawToDigi', 'rlxTaus'),
            []
        ),
        Test(
            'Checking Iso Taus',
            'BXVector<l1t::Tau>',
            ('caloStage1FinalDigis', 'isoTaus'),
            ('l1tRawToDigi', 'isoTaus'),
            []
        )
]

for event in events:
    print "< New event"
    for test in run:
        test(event)
