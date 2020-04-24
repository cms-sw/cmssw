#!/usr/bin/env python

import itertools
import unittest
import sys
import dataLoader
import ROOT


data = None
check_flavor = True
check_op = True
check_sys = True
verbose = False


def _eta_pt_discr_entries_generator(filter_keyfunc, op):
    assert data
    entries = filter(filter_keyfunc, data.entries)

    # use full or half eta range?
    if any(e.params.etaMin < 0. for e in entries):
        eta_test_points = data.eta_test_points
    else:
        eta_test_points = data.abseta_test_points

    for eta in eta_test_points:
        for pt in data.pt_test_points:
            ens_pt_eta = filter(
                lambda e:
                e.params.etaMin < eta < e.params.etaMax and
                e.params.ptMin < pt < e.params.ptMax,
                entries
            )
            if op == 3:
                for discr in data.discr_test_points:
                    ens_pt_eta_discr = filter(
                        lambda e:
                        e.params.discrMin < discr < e.params.discrMax,
                        ens_pt_eta
                    )
                    yield eta, pt, discr, ens_pt_eta_discr
            else:
                yield eta, pt, None, ens_pt_eta


class BtagCalibConsistencyChecker(unittest.TestCase):
    def test_lowercase(self):
        for item in [data.meas_type] + list(data.syss):
            self.assertEqual(
                item, item.lower(),
                "Item is not lowercase: %s" % item
            )

    def test_ops_tight(self):
        if check_op:
            self.assertIn(2, data.ops, "OP_TIGHT is missing")

    def test_ops_medium(self):
        if check_op:
            self.assertIn(1, data.ops, "OP_MEDIUM is missing")

    def test_ops_loose(self):
        if check_op:
            self.assertIn(0, data.ops, "OP_LOOSE is missing")

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

    def test_systematics_name(self):
        if check_sys:
            for syst in data.syss:
                if syst == 'central':
                    continue
                self.assertTrue(
                    syst.startswith("up") or syst.startswith("down"),
                    "sys. uncert name must start with 'up' or 'down' : %s"
                    % syst
                )

    def test_systematics_doublesidedness(self):
        if check_sys:
            for syst in data.syss:
                if "up" in syst:
                    other = syst.replace("up", "down")
                    self.assertIn(other, data.syss,
                                  "'%s' sys. uncert. is missing" % other)
                elif "down" in syst:
                    other = syst.replace("down", "up")
                    self.assertIn(other, data.syss,
                                  "'%s' sys. uncert. is missing" % other)

    def test_systematics_values_vs_centrals(self):
        if check_sys:
            res = list(itertools.chain.from_iterable(
                self._check_sys_side(op, flav)
                for flav in data.flavs
                for op in data.ops
            ))
            self.assertFalse(bool(res), "\n"+"\n".join(res))

    def _check_sys_side(self, op, flav):
        region = "op=%d, flav=%d" % (op, flav)
        if verbose:
            print "Checking sys side correctness for", region

        res = []
        for eta, pt, discr, entries in _eta_pt_discr_entries_generator(
            lambda e:
            e.params.operatingPoint == op and
            e.params.jetFlavor == flav,
            op
        ):
            if not entries:
                continue

            for e in entries:  # do a little monkey patching with tf1's
                if not hasattr(e, 'tf1_func'):
                    e.tf1_func = ROOT.TF1("", e.formula)

            sys_dict = dict((e.params.sysType, e) for e in entries)
            assert len(sys_dict) == len(entries)
            sys_cent = sys_dict.pop('central', None)
            x = discr if op == 3 else pt
            for syst, e in sys_dict.iteritems():
                sys_val = e.tf1_func.Eval(x)
                cent_val = sys_cent.tf1_func.Eval(x)
                if syst.startswith('up') and not sys_val > cent_val:
                    res.append(
                        ("Up variation '%s' not larger than 'central': %s "
                         "eta=%f, pt=%f " % (syst, region, eta, pt))
                        + ((", discr=%f" % discr) if discr else "")
                    )
                elif syst.startswith('down') and not sys_val < cent_val:
                    res.append(
                        ("Down variation '%s' not smaller than 'central': %s "
                         "eta=%f, pt=%f " % (syst, region, eta, pt))
                        + ((", discr=%f" % discr) if discr else "")
                    )
        return res

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
            self._check_coverage(op, syst, flav)
            for flav in data.flavs
            for syst in data.syss
            for op in data.ops
        ))
        self.assertFalse(bool(res), "\n"+"\n".join(res))

    def _check_coverage(self, op, syst, flav):
        region = "op=%d, %s, flav=%d" % (op, syst, flav)
        if verbose:
            print "Checking coverage for", region

        # walk over all testpoints
        res = []
        for eta, pt, discr, entries in _eta_pt_discr_entries_generator(
            lambda e:
            e.params.operatingPoint == op and
            e.params.sysType == syst and
            e.params.jetFlavor == flav,
            op
        ):
            size = len(entries)
            if size == 0:
                res.append(
                    ("Region not covered: %s eta=%f, pt=%f "
                     % (region, eta, pt))
                    + ((", discr=%f" % discr) if discr else "")
                )
            elif size > 1:
                res.append(
                    ("Region covered %d times: %s eta=%f, pt=%f"
                     % (size, region, eta, pt))
                    + ((", discr=%f" % discr) if discr else "")
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
    for dat in data_loaders:
        data = dat
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

