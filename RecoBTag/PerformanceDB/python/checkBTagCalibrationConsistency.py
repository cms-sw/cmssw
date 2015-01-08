import itertools
import unittest
import sys
import dataLoader


data = None
check_flavor = True
check_op = True
check_sys = True
verbose = False


class BtagCalibConsistencyChecker(unittest.TestCase):
    def test_ops_tight(self):
        if check_op:
            self.assertIn(0, data.ops, "OP_TIGHT is missing")

    def test_ops_medium(self):
        if check_op:
            self.assertIn(1, data.ops, "OP_MEDIUM is missing")

    def test_ops_loose(self):
        if check_op:
            self.assertIn(2, data.ops, "OP_LOOSE is missing")

    def test_flavs_b(self):
        if check_flavor:
            self.assertIn(0, data.flavs, "FLAV_B is missing")

    def test_flavs_c(self):
        if check_flavor:
            self.assertIn(1, data.flavs, "FLAV_C is missing")

    def test_flavs_udsg(self):
        if check_flavor:
            self.assertIn(2, data.flavs, "FLAV_UDSG is missing")

    def test_systematics_central(self):
        if check_sys:
            self.assertIn("central", data.syss,
                          "'central' sys. uncert. is missing")

    def test_systematics_up(self):
        if check_sys:
            self.assertIn("up", data.syss, "'up' sys. uncert. is missing")

    def test_systematics_down(self):
        if check_sys:
            self.assertIn("down", data.syss, "'down' sys. uncert. is missing")

    def test_systematics_doublesidedness(self):
        if check_sys:
            for sys in data.syss:
                if "up" in sys:
                    other = sys.replace("up", "down")
                    self.assertIn(other, data.syss,
                                  "'%s' sys. uncert. is missing" % other)
                elif "down" in sys:
                    other = sys.replace("down", "up")
                    self.assertIn(other, data.syss,
                                  "'%s' sys. uncert. is missing" % other)

    def test_eta_ranges(self):
        for a, b in data.etas:
            self.assertLess(a, b)
            self.assertGreater(a, data.ETA_MIN - 1e-7)
            self.assertLess(b, data.ETA_MAX + 1e-7)

    def test_pt_ranges(self):
        for a, b in data.pts:
            self.assertLess(a, b)
            self.assertGreater(a, data.PT_MIN - 1e-7)
            self.assertLess(b, data.PT_MAX + 1e-7)

    def test_discr_ranges(self):
        for a, b in data.discrs:
            self.assertLess(a, b)
            self.assertGreater(a, data.DISCR_MIN - 1e-7)
            self.assertLess(b, data.DISCR_MAX + 1e-7)

    def test_coverage(self):
        res = list(itertools.chain.from_iterable(
            self._check_coverage(op, sys, flav)
            for flav in data.flavs
            for sys in data.syss
            for op in data.ops
        ))
        self.assertFalse(bool(res), "\n"+"\n".join(res))

    def _check_coverage(self, op, sys, flav):
        region = "op=%d, %s, flav=%d" % (op, sys, flav)
        if verbose:
            print "Checking coverage for", region

        # load relevant entries
        ens = filter(
            lambda e:
            e.params.operatingPoint == op and
            e.params.sysType == sys and
            e.params.jetFlavor == flav,
            data.entries
        )

        # use full or half eta range?
        if any(e.params.etaMin < 0. for e in ens):
            eta_test_points = data.eta_test_points
        else:
            eta_test_points = data.abseta_test_points

        # walk over all testpoints
        res = []
        for eta in eta_test_points:
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


def run_check(filename, op=True, sys=True, flavor=True):
    loaders = dataLoader.get_data(filename)
    return run_check_data(loaders, op, sys, flavor)


def run_check_csv(csv_data, op=True, sys=True, flavor=True):
    loaders = dataLoader.get_data_csv(csv_data)
    return run_check_data(loaders, op, sys, flavor)


def run_check_data(data_loaders,
                   op=True, sys=True, flavor=True):
    global data, check_op, check_sys, check_flavor
    check_op, check_sys, check_flavor = op, sys, flavor

    all_res = []
    for data in data_loaders:
        print '\n\n'
        print '# Checking csv data for type / op / flavour:', \
            data.meas_type, data.op, data.flav
        print '='*60 + '\n'
        if verbose:
            data.print_data()
        testsuite = unittest.TestLoader().loadTestsFromTestCase(
            BtagCalibConsistencyChecker)
        res = unittest.TextTestRunner().run(testsuite)
        all_res.append(not bool(res.failures))
    return all_res


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Need csv data file as first argument.'
        print 'Options:'
        print '    --light (do not check op, sys, flav)'
        print '    --separate-by-op'
        print '    --separate-by-flav'
        print '    --separate-all (both of the above)'
        print 'Exit.'
        exit(-1)

    ck_op = ck_sy = ck_fl = not '--light' in sys.argv

    dataLoader.separate_by_op   = '--separate-by-op'   in sys.argv
    dataLoader.separate_by_flav = '--separate-by-flav' in sys.argv

    if '--separate-all' in sys.argv:
        dataLoader.separate_by_op = dataLoader.separate_by_flav = True

    if dataLoader.separate_by_op:
        ck_op = False
    if dataLoader.separate_by_flav:
        ck_fl = False

    verbose = True
    if not all(run_check(sys.argv[1], ck_op, ck_sy, ck_fl)):
        exit(-1)

