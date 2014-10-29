import itertools
import unittest
import sys
import ROOT
ROOT.gSystem.Load('libCondFormatsBTagObjects')


ETA_MIN = -2.4
ETA_MAX = 2.4
PT_MIN = 20.
PT_MAX = 1000.
DISCR_MIN = 0.
DISCR_MAX = 999.


class DataLoader(object):
    def __init__(self, csv_data):

        # list of entries
        ens = list(ROOT.BTagEntry(l) for l in csv_data)
        self.entries = ens

        # sets of fixed data
        self.ops = set(e.params.operatingPoint for e in ens)
        self.flavs = set(e.params.jetFlavor for e in ens)
        self.meass = set(e.params.measurementType for e in ens)
        self.syss = set(e.params.sysType for e in ens)
        self.etas = set((e.params.etaMin, e.params.etaMax) for e in ens)
        self.pts = set((e.params.ptMin, e.params.ptMax) for e in ens)
        self.discrs = set((e.params.discrMin, e.params.discrMax) for e in ens)

        self.full_eta_mode = any(eta_min < 0. for eta_min, _ in self.etas)

        # test points for variable data (using bound +- epsilon)
        eps = 1e-4
        eta_min = ETA_MIN if self.full_eta_mode else 0.
        self.eta_test_points = list(itertools.ifilter(
            lambda x: eta_min < x < ETA_MAX,
            itertools.chain(
                (a + eps for a, _ in self.etas),
                (a - eps for a, _ in self.etas),
                (b + eps for _, b in self.etas),
                (b - eps for _, b in self.etas),
                (eta_min + eps, ETA_MAX - eps),
            )
        ))
        self.pt_test_points = list(itertools.ifilter(
            lambda x: PT_MIN < x < PT_MAX,
            itertools.chain(
                (a + eps for a, _ in self.pts),
                (a - eps for a, _ in self.pts),
                (b + eps for _, b in self.pts),
                (b - eps for _, b in self.pts),
                (PT_MIN + eps, PT_MAX - eps),
            )
        ))
        self.discr_test_points = list(itertools.ifilter(
            lambda x: DISCR_MIN < x < DISCR_MAX,
            itertools.chain(
                (a + eps for a, _ in self.discrs),
                (a - eps for a, _ in self.discrs),
                (b + eps for _, b in self.discrs),
                (b - eps for _, b in self.discrs),
                (DISCR_MIN + eps, DISCR_MAX - eps),
            )
        ))
        # use sets
        self.eta_test_points = set(round(f, 5) for f in self.eta_test_points)
        self.pt_test_points = set(round(f, 5) for f in self.pt_test_points)
        self.discr_test_points = set(round(f, 5) for f in self.discr_test_points)



        print "\nFound operating points (need at least 0, 1, 2):"
        print self.ops

        print "\nFound jet flavors (need 0, 1, 2):"
        print self.flavs

        print "\nFound measurement types (at least 'comb'):"
        print self.meass

        print "\nFound sys types (need at least 'central', 'up', 'down'; " \
              "also 'up_SYS'/'down_SYS' compatibility is checked):"
        print self.syss

        print "\nFound eta ranges: (need everything covered from %g or 0. " \
              "up to %g):" % (ETA_MIN, ETA_MAX)
        print self.etas

        print "\nFound pt ranges: (need everything covered from %g " \
              "to %g):" % (PT_MIN, PT_MAX)
        print self.pts

        print "\nFound discr ranges: (only needed for operatingPoint==3, " \
              "covered from %g to %g):" % (DISCR_MIN, DISCR_MAX)
        print self.discrs

        print "\nTest points for eta:"
        print self.eta_test_points

        print "\nTest points for pt:"
        print self.pt_test_points

        print "\nTest points for discr:"
        print self.discr_test_points
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

    def test_eta_ranges(self):
        for a, b in data.etas:
            self.assertLess(a, b)
            self.assertGreater(a, ETA_MIN - 1e-7)
            self.assertLess(b, ETA_MAX + 1e-7)

    def test_pt_ranges(self):
        for a, b in data.pts:
            self.assertLess(a, b)
            self.assertGreater(a, PT_MIN - 1e-7)
            self.assertLess(b, PT_MAX + 1e-7)

    def test_discr_ranges(self):
        for a, b in data.discrs:
            self.assertLess(a, b)
            self.assertGreater(a, DISCR_MIN - 1e-7)
            self.assertLess(b, DISCR_MAX + 1e-7)

    def test_coverage(self):
        res = list(itertools.chain.from_iterable(
            self._check_coverage(op, meas, sys, flav)
            for flav in data.flavs
            for sys in data.syss
            for meas in data.meass
            for op in data.ops
        ))
        self.assertFalse(bool(res), "\n"+"\n".join(res))

    def _check_coverage(self, op, meas, sys, flav):
        region = "op=%d, %s, %s, flav=%d" % (op, meas, sys, flav)
        print "Checking coverage for", region
        ens = filter(
            lambda e:
            e.params.operatingPoint == op and
            e.params.measurementType == meas and
            e.params.sysType == sys and
            e.params.jetFlavor == flav,
            data.entries
        )
        res = []
        for eta in data.eta_test_points:
            for pt in data.pt_test_points:
                tmp_eta_pt = filter(
                    lambda e:
                    e.params.etaMin < eta < e.params.etaMax and
                    e.params.ptMin < pt < e.params.ptMax,
                    ens
                )
                if op == 3:
                    for discr in data.discr_test_points:
                        tmp_eta_pt_discr = filter(
                            lambda e:
                            e.params.discrMin < discr < e.params.discrMax,
                            tmp_eta_pt
                        )
                        size = len(tmp_eta_pt_discr)
                        if size == 0:
                            res.append(
                                "Region not covered: %s eta=%f, pt=%f, "
                                "discr=%f" % (region, eta, pt, discr)
                            )
                        elif size > 1:
                            res.append(
                                "Region covered %d times: %s eta=%f, pt=%f, "
                                "discr=%f" % (size, region, eta, pt, discr)
                            )
                else:
                    size = len(tmp_eta_pt)
                    if size == 0:
                        res.append(
                            "Region not covered: "
                            "%s eta=%f, pt=%f" % (region, eta, pt)
                        )
                    elif size > 1:
                        res.append(
                            "Region covered %d times: "
                            "%s eta=%f, pt=%f" % (size, region, eta, pt)
                        )
        return res


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Need csv data file as first argument. Exit."
        exit(-1)
    with open(sys.argv.pop(1)) as f:
        lines = f.readlines()
        if not (lines and "OperatingPoint" in lines[0]):
            print "Data file does not contain typical header. Exit."
            exit(-1)
        lines.pop(0)  # remove header
        data = DataLoader(lines)
    unittest.main()

