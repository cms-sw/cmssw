import os
import sys
import itertools
import checkBTagCalibrationConsistency as checker


def generate_flav_c(loaded_data):
    flav_b_data = filter(
        lambda e: e.params.jetFlavor == 0,
        loaded_data.entries
    )
    flav_b_data = sorted(flav_b_data, key=lambda e: e.params.operatingPoint)
    flav_b_data = sorted(flav_b_data, key=lambda e: e.params.measurementType)
    flav_b_data = sorted(flav_b_data, key=lambda e: e.params.etaMin)
    flav_b_data = sorted(flav_b_data, key=lambda e: e.params.ptMin)
    flav_b_data = sorted(flav_b_data, key=lambda e: e.params.discrMin)
    sys_groups = itertools.groupby(flav_b_data,
                                   key=lambda e: e.params.operatingPoint)

    def gen_entry_dict(groups):
        for _, grp in groups:
            grp = list(grp)
            entries_by_sys = dict((e.params.sysType, e) for e in grp)
            assert len(grp) == len(entries_by_sys)  # every sysType is unique
            yield entries_by_sys
    sys_dicts = gen_entry_dict(sys_groups)

    def gen_flavb_csv_line(dicts):
        for d in dicts:
            central = d.pop('central')
            central.params.jetFlavor = 1
            yield central.makeCSVLine()
            for e in d.itervalues():
                e.params.jetFlavor = 1
                e.formula = '2*(%s)-(%s)' % (e.formula, central.formula)
                yield e.makeCSVLine()
    csv_lines = gen_flavb_csv_line(sys_dicts)

    return list(l for l in csv_lines)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Need input/output filenames as first/second arguments. Exit.'
        exit(-1)
    if os.path.exists(sys.argv[2]):
        print 'Output file exists. Exit.'
        exit(-1)

    print '\nChecking input file consistency...'
    if not checker.run_check(sys.argv[1], True, True, False, False):
        print 'Checks on input file failed. Exit.'
        exit(-1)
    if not 0 in checker.data.flavs:
        print 'FLAV_B not found in input file. Exit.'
        exit(-1)
    if 1 in checker.data.flavs:
        print 'FLAV_C already present in input file. Exit.'
        exit(-1)

    print '\nGenerating new csv content...'
    new_csv_data = generate_flav_c(checker.data)

    with open(sys.argv[1]) as f:
        old_csv_data = f.readlines()

    with open(sys.argv[2], 'w') as f:
        f.writelines(old_csv_data)
        f.write('\n\n')
        f.writelines(new_csv_data)

    print 'Done.'

