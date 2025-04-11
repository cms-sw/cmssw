import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pdb import set_trace
eps = 10**-7
from argparse import ArgumentParser
import numpy as np

def writer(pct):
  if pct < 5: return ''
  else: return '%.1f%%' % pct

from collections import defaultdict
import matplotlib.colors as colors
import matplotlib.cm as cmx

parser = ArgumentParser()
parser.add_argument('invals', nargs='+', help='file path : name to use')
args = parser.parse_args()

tags_timings = []
for inval in args.invals:
  #Module Summary
  fname, tag = tuple(inval.split(':'))
  infile = open(fname).read()
  modules = infile.split('TimeReport ---------- Module Summary ---[Real sec]----')[1].split('T---Report end!')[0]
  time_rep = infile.split('TimeReport ---------- Event  Summary')[1].split('\n\n')[0].split('\n')
  
  module_times = []
  for l in modules.split('\n'):
    line = l.strip()
    if not line: continue
    if line.endswith('Name'): continue
    info = line.split()
    module_times.append((info[-1], float(info[1])))
  
  tot_time = sum(i for _, i in module_times)
  tags_timings.append((tag, tot_time, dict(module_times)))
  groups = defaultdict(float)
  
  for name, time in module_times:
    if 'gen' in name.lower():
      groups['GEN'] += time
    elif 'lhe' in name.lower():
      groups['GEN'] += time
    elif 'Table' in name:
      groups['Tables (not GEN)'] += time
    elif 'kee' in name.lower():
      groups['BToKee'] += time
    elif 'kmumu' in name.lower():
      groups['BToKmumu'] += time
    elif 'electron' in name.lower():
      groups['Electrons'] += time
    elif 'track' in name.lower():
      groups['Tracks'] += time
    elif 'muon' in name.lower():
      groups['Muons'] += time
    elif name.endswith('output'):
      groups['I/O'] += time
    else:
      groups['Other'] += time
  
  cm = plt.get_cmap('rainbow')
  cNorm  = colors.Normalize(vmin=0, vmax=len(groups)-1)
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
  cols = [scalarMap.to_rgba(i) for i in range(len(groups))]
  
  
  plt.clf()
  fig = plt.figure(
      figsize=(12, 6), 
  )
  plt.subplot(1, 2, 1)
  vals = np.array(groups.values())/tot_time
  wedges = plt.pie(vals, autopct = writer, colors = cols)
  names = ['%s (%.1f%%)' % (n, float(p)*100/tot_time) for n, p in groups.iteritems()]
  leg = plt.legend(
      wedges[0], names, loc = 5,
      bbox_to_anchor = (0.95, 0.5),
      mode="expand", borderaxespad=0., frameon=False
  )
  cpu = float(time_rep[1].split(' = ')[1])
  wall = float(time_rep[2].split(' = ')[1])
  
  title = '%s Execution time: %.3f [s] (CPU / evt), %.3f [s] (Wall / evt)' % (tag, cpu, wall)
  plt.title(title)
  fig.savefig('validation/timing_%s.png' % tag)

table = []
# get the max size of the module name
max_name = max(
  max(
    map(lambda x: len(x), i.keys())
  ) for _, _, i in tags_timings
)
name_format = '%-'+str(max_name)+'s'

# get the max size of the tag name, the float takes minimum 6 chars (0.1234)
max_tag = max(
  max(len(i) for i, _, _ in tags_timings),
  6
)
tag_format = '%'+str(max_tag)+'s'

tot_size = max_name + len(tags_timings)*(max_tag+3) + 4
line = '-'*tot_size+'\n'

def make_line(name, vals):
  formatted_name = name_format % name
  formatted_vals = [tag_format % i for i in vals]
  line = ' | '.join([formatted_name] + formatted_vals)
  return '| %s |\n' % line

all_modules = set()
for _, _, i in tags_timings:
  all_modules.update(i.keys())

with open('validation/timing.txt', 'w') as out:
  out.write(line)
  out.write(
    make_line('Module', [i for i, _, _ in tags_timings])
    )
  out.write(line)
  for mod in all_modules:
    vals = [
      '%.4f' % i[mod] if mod in i else '   -- ' 
      for _, _, i in tags_timings
    ]
    out.write(
      make_line(mod, vals)
      )
  out.write(line)
  
