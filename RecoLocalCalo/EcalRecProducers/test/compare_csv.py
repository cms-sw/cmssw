from __future__ import print_function
import sys

def compare(fn1, fn2):
    f1, f2 = open(fn1, "r"), open(fn2, "r")
    diff_cols = {}

    for l1, l2 in zip(f1.readlines(), f2.readlines()):
        if l1 == l2: continue

        for i, (k1, k2) in enumerate(zip(l1.strip().split(), l2.strip().split())):
            if k1 == k2: continue

            if i not in diff_cols:
                diff_cols[i] = []

            diff = diff_cols[i]

            try:
                diff_f = abs(float(k2) - float(k1))
                diff.append((diff_f, k1, k2, "l1: " + l1, "l2: " + l2, ))

                print("diffrence[f%d]: %s -> %s" % (i, k1, k2))
            except ValueError:
                print("non float-type difference[f%d]: %s -> %s" % (i, k1, k2))

    for key, item in sorted(diff_cols.items()):
        print("column:", key)
        print("\tavg: %f" % (sum(map(lambda x: x[0], item)) / len(item)))

        m = max(item)
        print("\tmax:", m[:3])
        print("\t\t-:", m[3])
        print("\t\t+:", m[4])
        
    return diff_cols

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage %s old.csv new.csv" % sys.argv[0])
        sys.exit(1)

    diff = compare(sys.argv[1], sys.argv[2])
