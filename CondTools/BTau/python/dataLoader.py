import itertools
import ROOT
try:
    ROOT.BTagEntry
except AttributeError:
    ROOT.gROOT.ProcessLine('.L BTagCalibrationStandalone.cpp+')

try:
    ROOT.BTagEntry
except AttributeError:
    print 'ROOT.BTagEntry is needed! Please copy ' \
          'BTagCalibrationStandalone.[h|cpp] to the working directory. Exit.'
    exit(-1)

separate_by_op   = False
separate_by_flav = False


class DataLoader(object):
    def __init__(self, csv_data, measurement_type, operating_point, flavour):
        self.meas_type = measurement_type
        self.op = operating_point
        self.flav = flavour

        # list of entries
        ens = []
        for l in csv_data:
            if not l.strip():
                continue  # skip empty lines
            try:
                e = ROOT.BTagEntry(l)
                if (e.params.measurementType == measurement_type
                    and ((not separate_by_op)
                            or e.params.operatingPoint == operating_point)
                    and ((not separate_by_flav)
                            or e.params.jetFlavor == flavour)
                ):
                    ens.append(e)
            except TypeError:
                raise RuntimeError("Error: can not interpret line: " + l)
        self.entries = ens

        if not ens:
            return

        # fixed data
        self.ops = set(e.params.operatingPoint for e in ens)
        self.flavs = set(e.params.jetFlavor for e in ens)
        self.syss = set(e.params.sysType for e in ens)
        self.etas = set((e.params.etaMin, e.params.etaMax) for e in ens)
        self.pts = set((e.params.ptMin, e.params.ptMax) for e in ens)
        self.discrs = set((e.params.discrMin, e.params.discrMax)
                          for e in ens
                          if e.params.operatingPoint == 3)

        self.ETA_MIN = -2.4
        self.ETA_MAX = 2.4
        self.PT_MIN = min(e.params.ptMin for e in ens)
        self.PT_MAX = max(e.params.ptMax for e in ens)
        if any(e.params.operatingPoint == 3 for e in ens):
            self.DISCR_MIN = min(
                e.params.discrMin
                for e in ens
                if e.params.operatingPoint == 3
            )
            self.DISCR_MAX = max(
                e.params.discrMax
                for e in ens
                if e.params.operatingPoint == 3
            )
        else:
            self.DISCR_MIN = 0.
            self.DISCR_MAX = 1.

        # test points for variable data (using bound +- epsilon)
        eps = 1e-4
        eta_test_points = list(itertools.ifilter(
            lambda x: self.ETA_MIN < x < self.ETA_MAX,
            itertools.chain(
                (a + eps for a, _ in self.etas),
                (a - eps for a, _ in self.etas),
                (b + eps for _, b in self.etas),
                (b - eps for _, b in self.etas),
                (self.ETA_MIN + eps, self.ETA_MAX - eps),
            )
        ))
        abseta_test_points = list(itertools.ifilter(
            lambda x: 0. < x < self.ETA_MAX,
            itertools.chain(
                (a + eps for a, _ in self.etas),
                (a - eps for a, _ in self.etas),
                (b + eps for _, b in self.etas),
                (b - eps for _, b in self.etas),
                (eps, self.ETA_MAX - eps),
            )
        ))
        pt_test_points = list(itertools.ifilter(
            lambda x: self.PT_MIN < x < self.PT_MAX,
            itertools.chain(
                (a + eps for a, _ in self.pts),
                (a - eps for a, _ in self.pts),
                (b + eps for _, b in self.pts),
                (b - eps for _, b in self.pts),
                (self.PT_MIN + eps, self.PT_MAX - eps),
            )
        ))
        discr_test_points = list(itertools.ifilter(
            lambda x: self.DISCR_MIN < x < self.DISCR_MAX,
            itertools.chain(
                (a + eps for a, _ in self.discrs),
                (a - eps for a, _ in self.discrs),
                (b + eps for _, b in self.discrs),
                (b - eps for _, b in self.discrs),
                (self.DISCR_MIN + eps, self.DISCR_MAX - eps),
            )
        ))
        # use sets
        self.eta_test_points = set(round(f, 5) for f in eta_test_points)
        self.abseta_test_points = set(round(f, 5) for f in abseta_test_points)
        self.pt_test_points = set(round(f, 5) for f in pt_test_points)
        self.discr_test_points = set(round(f, 5) for f in discr_test_points)

    def print_data(self):
        print "\nFound operating points:"
        print self.ops

        print "\nFound jet flavors:"
        print self.flavs

        print "\nFound sys types (need at least 'central', 'up', 'down'; " \
              "also 'up_SYS'/'down_SYS' compatibility is checked):"
        print self.syss

        print "\nFound eta ranges: (need everything covered from %g or 0. " \
              "up to %g):" % (self.ETA_MIN, self.ETA_MAX)
        print self.etas

        print "\nFound pt ranges: (need everything covered from %g " \
              "to %g):" % (self.PT_MIN, self.PT_MAX)
        print self.pts

        print "\nFound discr ranges: (only needed for operatingPoint==3, " \
              "covered from %g to %g):" % (self.DISCR_MIN, self.DISCR_MAX)
        print self.discrs

        print "\nTest points for eta (bounds +- epsilon):"
        print self.eta_test_points

        print "\nTest points for pt (bounds +- epsilon):"
        print self.pt_test_points

        print "\nTest points for discr (bounds +- epsilon):"
        print self.discr_test_points
        print ""


def get_data_csv(csv_data):
    # grab measurement types
    meas_types = set(
        l.split(',')[1].strip()
        for l in csv_data
        if len(l.split()) == 11
    )

    # grab operating points
    ops = set(
        int(l.split(',')[0])
        for l in csv_data
        if len(l.split()) == 11
    ) if separate_by_op else ['all']

    # grab flavors
    flavs = set(
        int(l.split(',')[3])
        for l in csv_data
        if len(l.split()) == 11
    ) if separate_by_flav else ['all']

    # make loaders and filter empty ones
    lds = list(
        DataLoader(csv_data, mt, op, fl)
        for mt in meas_types
        for op in ops
        for fl in flavs
    )
    lds = filter(lambda d: d.entries, lds)
    return lds


def get_data(filename):
    with open(filename) as f:
        csv_data = f.readlines()
    if not (csv_data and "OperatingPoint" in csv_data[0]):
        print "Data file does not contain typical header: %s. Exit" % filename
        return False
    csv_data.pop(0)  # remove header
    return get_data_csv(csv_data)
