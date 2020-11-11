import os
import re

#Online
entries=os.listdir('../')

for entry in entries:
	print(entry)
	LAYOUTS = re.findall("[^-_]*-layouts.py", entry)
	LAYOUTS += re.findall("shift_[^-_]*_layout.py",entry)
	LAYOUTS += re.findall(".*_overview_layouts.py", entry)
	LAYOUTS = re.findall('AAAA', "A")
print(LAYOUTS)
