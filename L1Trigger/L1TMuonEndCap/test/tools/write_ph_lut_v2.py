#!/usr/bin/env python

import os

in_path = 'L1Trigger/L1TMuonEndCap/test/tools/pc_luts/firmware_data/'
# in_path = 'L1Trigger/L1TMuonEndCap/test/tools/pc_luts/firmware_MC/'

def main():
  full_path = os.environ['CMSSW_BASE'] + '/src/' + in_path + '%s'
  out_dir   = os.environ['CMSSW_BASE'] + '/src/' + in_path.replace('firmware', 'emulator')

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  all_files = []

  ph_init_array = []
  ph_disp_array = []
  th_init_array = []
  th_disp_array = []
  th_lut_array = []
  th_corr_lut_array = []

  # ____________________________________________________________________________
  def read_file_into_array(fn, arr, resize=0):
    old_size = len(arr)
    with open(fn) as f:
      for line in f:
        for s in line.split():
          x = int(s, 16)
          arr.append(x)

    new_size = len(arr)
    if resize:
      for i in xrange(new_size-old_size, resize):
        arr.append(0)
    return

  def dump_array_into_file(arr, fn):
    with open(fn, 'w') as f:
      s = ""
      for i in xrange(len(arr)):
        x = arr[i]
        s += ("%i " % x)
        if (i+1)%30 == 0:
          s += "\n"
      f.write(s)

  # ____________________________________________________________________________
  for endcap in [1,2]:
    for sector in [1,2,3,4,5,6]:

      # ph_init
      for st in [0,1,2,3,4]:
        ph_init_filename = 'ph_init_full_endcap_%i_sect_%i_st_%i.lut' % (endcap, sector, st)
        read_file_into_array((full_path % ph_init_filename), ph_init_array)
        all_files.append(ph_init_filename)

      # ph_disp
      ph_disp_filename = 'ph_disp_endcap_%i_sect_%i.lut' % (endcap, sector)
      read_file_into_array((full_path % ph_disp_filename), ph_disp_array)
      all_files.append(ph_disp_filename)

      # th_init
      th_init_filename = 'th_init_endcap_%i_sect_%i.lut' % (endcap, sector)
      read_file_into_array((full_path % th_init_filename), th_init_array)
      all_files.append(th_init_filename)

      # th_disp
      th_disp_filename = 'th_disp_endcap_%i_sect_%i.lut' % (endcap, sector)
      read_file_into_array((full_path % th_disp_filename), th_disp_array)
      all_files.append(th_disp_filename)

      # th_lut
      for sub in [1]:
        for st in [1]:
          for ch in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
            th_lut_filename = 'vl_th_lut_endcap_%i_sec_%i_sub_%i_st_%i_ch_%i.lut' % (endcap, sector, sub, st, ch)
            read_file_into_array((full_path % th_lut_filename), th_lut_array, resize=128)
            all_files.append(th_lut_filename)

      for sub in [2]:
        for st in [1]:
          for ch in [1,2,3,4,5,6,7,8,9,10,11,12]:
            th_lut_filename = 'vl_th_lut_endcap_%i_sec_%i_sub_%i_st_%i_ch_%i.lut' % (endcap, sector, sub, st, ch)
            read_file_into_array((full_path % th_lut_filename), th_lut_array, resize=128)
            all_files.append(th_lut_filename)

      for st in [2,3,4]:
        for ch in [1,2,3,4,5,6,7,8,9,10,11]:
          th_lut_filename = 'vl_th_lut_endcap_%i_sec_%i_st_%i_ch_%i.lut' % (endcap, sector, st, ch)
          read_file_into_array((full_path % th_lut_filename), th_lut_array, resize=128)
          all_files.append(th_lut_filename)

      # th_corr_lut
      for sub in [1]:
        for st in [1]:
          for ch in [1,2,3,13]:
            th_corr_lut_filename = 'vl_th_corr_lut_endcap_%i_sec_%i_sub_%i_st_%i_ch_%i.lut' % (endcap, sector, sub, st, ch)
            read_file_into_array((full_path % th_corr_lut_filename), th_corr_lut_array, resize=128)
            all_files.append(th_corr_lut_filename)

      for sub in [2]:
        for st in [1]:
          for ch in [1,2,3]:
            th_corr_lut_filename = 'vl_th_corr_lut_endcap_%i_sec_%i_sub_%i_st_%i_ch_%i.lut' % (endcap, sector, sub, st, ch)
            read_file_into_array((full_path % th_corr_lut_filename), th_corr_lut_array, resize=128)
            all_files.append(th_corr_lut_filename)

      pass  # end loop over sector
    pass  # end loop over endcap

  # ____________________________________________________________________________
  assert(len(all_files) == 12*76)
  assert(len(ph_init_array) == 12*61)
  assert(len(ph_disp_array) == 12*61)
  assert(len(th_init_array) == 12*61)
  assert(len(th_disp_array) == 12*61)
  assert(len(th_lut_array) == 12*61*128)
  assert(len(th_corr_lut_array) == 12*7*128)

  dump_array_into_file(ph_init_array, out_dir+"ph_init_neighbor.txt")
  dump_array_into_file(ph_disp_array, out_dir+"ph_disp_neighbor.txt")
  dump_array_into_file(th_init_array, out_dir+"th_init_neighbor.txt")
  dump_array_into_file(th_disp_array, out_dir+"th_disp_neighbor.txt")
  dump_array_into_file(th_lut_array, out_dir+"th_lut_neighbor.txt")
  dump_array_into_file(th_corr_lut_array, out_dir+"th_corr_lut_neighbor.txt")

  return


# ______________________________________________________________________________
if __name__ == '__main__':

  main()
