filename = "basic_lats.txt"

with open(filename) as f:
    lines = f.readlines()

for iline, line in enumerate(lines):
    line = line.strip().split("/")
    # text = "if      " if iline == 0 else "else if "
    # text += "(inMPath->missingLayer() == %s" % line[0]
    # text += " && are_equal(inMPath->cellLayout(), (int[4])(%s))) " % line[1]
    # text += "lateralities.push_back(%s);" % line[2]
    # print(text)
    text = "lat_combinations.push_back({ %s, %s, %s });" % (line[0], line[1], line[2])
    print(text)