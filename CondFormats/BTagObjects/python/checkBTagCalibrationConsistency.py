import unittest
import ROOT
ROOT.gSystem.Load('libCondFormatsBTagObjects')


def get_csv_data():
    return [
        "0, comb, central, 0, 0, 1, 0, 1, 0, 999, \"2*x\" \n",
        "0, comb, central, 0, 0, 1, 1, 2, 0, 999, \"2*x\" \n",
        "0, comb, central, 0, 1, 2, 0, 1, 0, 999, \"-2*x\" \n",
        "0, comb, central, 0, 1, 2, 1, 2, 0, 999, \"-2*x\" \n",
        "3, comb, central, 0, 0, 1, 0, 1, 2, 3, \"2*x\" \n",
        "3, comb, central, 0, -1, 0, 0, 1, 2, 3, \"-2*x\" \n",
    ]


class DataLoader(object):
    def __init__(self, csv_data):
        ens = list(ROOT.BTagEntry(l) for l in csv_data)
        self.ops = set(e.params.operatingPoint for e in ens)
        self.flavs = set(e.params.jetFlavor for e in ens)
        self.meass = set(e.params.measurementType for e in ens)
        self.syss = set(e.params.sysType for e in ens)
        self.etas = set((e.params.etaMin, e.params.etaMax) for e in ens)
        self.pts = set((e.params.ptMin, e.params.ptMax) for e in ens)
        self.discrs = set((e.params.discrMin, e.params.discrMax) for e in ens)
        self.entries = ens

        print "\nFound operating points (need at least 0, 1, 2):"
        print self.ops

        print "\nFound jet flavors (need 0, 1, 2):"
        print self.flavs

        print "\nFound measurement types (at least 'comb'):"
        print self.meass

        print "\nFound sys types (need at least 'central', 'up', 'down'; " \
              "also 'up_SYS'/'down_SYS' compatibility is checked):"
        print self.syss

        print "\nFound eta ranges: (need everything covered from -2.4 or 0. " \
              "up to 2.4):"
        print self.etas

        print "\nFound pt ranges: (need everything covered from 20. to 1000.):"
        print self.pts

        print "\nFound discr ranges: (only needed for operatingPoint==3, " \
              "covered from 0. to 999.):"
        print self.discrs
        print ""
data = None


class BtagCalibConsistencyChecker(unittest.TestCase):
    def __init__(self, *args, **kws):
        super(BtagCalibConsistencyChecker, self).__init__(*args, **kws)

    def test_ops(self):
        self.assertIn(0, data.ops)
        self.assertIn(1, data.ops)
        self.assertIn(2, data.ops)

    def test_flavs(self):
        self.assertIn(0, data.flavs)
        self.assertIn(1, data.flavs)
        self.assertIn(2, data.flavs)

    def test_meass(self):
        self.assertIn("comb", data.meass)

    def test_syss(self):
        self.assertIn("central", data.syss)
        self.assertIn("up", data.syss)
        self.assertIn("down", data.syss)
        for sys in data.syss:
            if "up" in sys:
                self.assertIn(sys.replace("up", "down"), data.syss)
            elif "down" in sys:
                self.assertIn(sys.replace("down", "up"), data.syss)


if __name__ == '__main__':
    global data
    data = DataLoader(get_csv_data())
    unittest.main()

