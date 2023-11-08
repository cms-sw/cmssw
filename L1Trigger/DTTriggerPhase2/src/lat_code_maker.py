filename = "basic_lats.txt"

with open(filename) as f:
    lines = f.readlines()

for iline, line in enumerate(lines):
    line = line.strip().split("/")
    text = "lat_combinations.push_back({ %s, %s, %s });" % (line[0], line[1], line[2])
    print(text)
