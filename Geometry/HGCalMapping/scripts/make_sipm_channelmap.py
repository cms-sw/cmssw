
def get_sequence_for_channel(channel : int, isHD : bool) -> int:
  """
  Determine the sequence number for a given channel on a tileboard,
  depending on whether it is a low-density (LD) or high-density (HD) tileboard.
  For 36 channels per ROC and halfROC, the returned sequence number will range from 0 to 35.

  Parameters:
  - channel (int): The channel number.
  - isHD (bool): True if the tileboard is HD, False if LD.

  Returns:
  - seq (int): The calculated sequence number for the channel within its ROC and halfROC grouping.
  """
  # for LD tileboards
  if not isHD:
    # assigning ROC and halfROC values based on channel number
    if(channel < 36):
      seq = channel
    elif(channel < 72):
      seq = channel - 36
    else:
      seq = channel - 72
  else:
   # for HD tileboards
    if(channel < 36):
      seq = channel
    elif(channel < 72):
      seq = channel - 36
    else:
      # channels are gives as channel number + 100
      channel_ = channel - 100
      if channel_ < 36:
        seq = channel_
      elif channel_ < 72:
        seq = channel_ - 36
  return seq

def get_global_ring_number(local_ring : int, ring_max : int) -> int:
  """
  Convert a local ring number into a global ring number for a module.

  Parameters:
  - local_ring (int): The ring number within a local coordinate system.
  - ring_max (int): The maximum ring number in the module.

  Returns:
  - ring (int): The global ring number, calculated as the inverted index from ring_max.  
  """
  ring = ring_max - local_ring
  return ring

def get_iphi_number(base_iphi : int, sector : int, isHD : bool) -> int:
  """
  Calculate the global iphi (phi index) number based on a base iphi, the sector, and tileboard density.
  Defined for 8 or 12 sectors for HD and LD tileboards, respectively.

  Parameters:
  - base_iphi (int): The base phi index within the sector.
  - sector (int): The sector number (1-based).
  - isHD (bool): True if HD tileboard, otherwise False.

  Returns:
  - iphi (int): The global iphi index.
  """
  niPhis = 8 if not isHD else 12
  iphi = base_iphi + (8 * (sector - 1))
  return iphi

def get_ring_max_and_n_rings(mod_n_rings : int, mod_ring_max : int, localROC : int, localHalfROC : int, hd_tiles : bool) -> tuple[int, int]:
  """
  Compute the maximum ring number and the number of rings for a given ROC and halfROC on a tileboard.

  Parameters:
  - mod_n_rings (int): Total number of rings in the module.
  - mod_ring_max (int): Maximum ring number for the module.
  - localROC (int): Local ROC number (e.g., 1 or 2).
  - localHalfROC (int): Local halfROC number (0 or 1).
  - hd_tiles (bool): True if the tileboard is HD, False otherwise.

  Returns:
  - ring_max (int): The adjusted maximum ring number based on ROC and halfROC.
  - n_rings (int): The number of rings assigned to this ROC and halfROC.
  """
  # 4 or 3 rings per halfROC for HD or LD tiles, respectively
  n_rings_roc1_halfroc0 = 4 if not hd_tiles else 3
  n_rings_roc1_halfroc1 = 8 if not hd_tiles else 6
  n_rings_roc2_halfroc0 = 12 if not hd_tiles else 9
  
  if localROC == 1:
    if localHalfROC == 1:
      ring_max = mod_ring_max
      n_rings = min(n_rings_roc1_halfroc0, mod_n_rings)
    else:
      ring_max = mod_ring_max - n_rings_roc1_halfroc0
      n_rings = mod_n_rings - n_rings_roc1_halfroc0
      if n_rings > n_rings_roc1_halfroc0: # for tileboards with more than 1 ROC
        n_rings = n_rings - n_rings_roc1_halfroc0
        if n_rings > n_rings_roc1_halfroc0:
          n_rings = n_rings - n_rings_roc1_halfroc0
  elif localROC == 2:
    if localHalfROC == 1:
      ring_max = mod_ring_max - n_rings_roc1_halfroc1
      n_rings = mod_n_rings - n_rings_roc1_halfroc1
      if n_rings > n_rings_roc1_halfroc0: # for tileboards with more than 1 ROC
        n_rings = n_rings - n_rings_roc1_halfroc0
        if n_rings > n_rings_roc1_halfroc0:
          n_rings = n_rings - n_rings_roc1_halfroc1
    else:
      ring_max = mod_ring_max - n_rings_roc2_halfroc0
      n_rings = mod_n_rings - n_rings_roc2_halfroc0
  
  return ring_max, n_rings

"""
Channel number mapping for SiPMs on tileboards.
From channel number to: (ring order, iphi, Trigger group and Trigger Channel).
Given for HD tileboards (8phi) and LD tileboards (12phi).
"""
# HD tileboards with 8 tiles / ring (8 iphi)
channel_number_to_channel_location_8phi = {
  # channel : (ring order, iphi, "Trig.-Out", "TC")
  105:  (11, 0, 6, 0),
  107:  (11, 1, 6, 0),
  101:  (11, 2, 6, 3),
  103:  (11, 3, 6, 3),
  96 :  (11, 4, 6, 2),
  98 :  (11, 5, 6, 2),
  92 :  (11, 6, 6, 1),
  94 :  (11, 7, 6, 1),
  89 :  (10,-1,-1,-1),
  90 :  (10,-1,-1,-1),
  104:  (10, 0, 6, 0),
  106:  (10, 1, 6, 0),
  100:  (10, 2, 6, 3),
  102:  (10, 3, 6, 3),
  99 :  (10,-1,-1,-1),
  95 :  (10, 4, 6, 2),
  97 :  (10, 5, 6, 2),
  91 :  (10, 6, 6, 1),
  93 :  (10, 7, 6, 1),
  82 :  (9,  0, 5, 3),
  84 :  (9,  1, 5, 3),
  77 :  (9,  2, 5, 2),
  79 :  (9,  3, 5, 2),
  73 :  (9,  4, 5, 0),
  75 :  (9,  5, 5, 0),
  86 :  (9,  6, 5, 1),
  88 :  (9,  7, 5, 1),
  81 :  (8,  0, 5, 3),
  83 :  (8,  1, 5, 3),
  80 :  (8, -1,-1,-1),
  76 :  (8,  2, 5, 2),
  78 :  (8,  3, 5, 2),
  72 :  (8,  4, 5, 0),
  74 :  (8,  5, 5, 0),
  85 :  (8,  6, 5, 1),
  87 :  (8,  7, 5, 1),
  
  69 :  (7,  0, 4, 3),
  71 :  (7,  1, 4, 3),
  65 :  (7,  2, 4, 2),
  67 :  (7,  3, 4, 2),
  60 :  (7,  4, 4, 1),
  62 :  (7,  5, 4, 1),
  56 :  (7,  6, 4, 0),
  58 :  (7,  7, 4, 0),
  68 :  (6,  0, 4, 3),
  70 :  (6,  1, 4, 3),
  63 :  (6, -1,-1,-1),
  64 :  (6,  2, 4, 2),
  66 :  (6,  3, 4, 2),
  59 :  (6,  4, 4, 1),
  61 :  (6,  5, 4, 1),
  53 :  (6, -1,-1,-1),
  54 :  (6, -1,-1,-1),
  55 :  (6,  6, 4, 0),
  57 :  (6,  7, 4, 0),
  41 :  (5,  0, 3, 1),
  43 :  (5,  1, 3, 1),
  37 :  (5,  2, 3, 0),
  39 :  (5,  3, 3, 0),
  50 :  (5,  4, 3, 3),
  52 :  (5,  5, 3, 3),
  46 :  (5,  6, 3, 2),
  48 :  (5,  7, 3, 2),
  40 :  (4,  0, 3, 1),
  42 :  (4,  1, 3, 1),
  36 :  (4,  2, 3, 0),
  38 :  (4,  3, 3, 0),
  49 :  (4,  4, 3, 3),
  51 :  (4,  5, 3, 3),
  44 :  (4, -1,-1,-1),
  45 :  (4,  6, 3, 2),
  47 :  (4,  7, 3, 2),
  20 :  (3,  0, 2, 0),
  22 :  (3,  1, 2, 0),
  33 :  (3,  2, 2, 3),
  35 :  (3,  3, 2, 3),
  29 :  (3,  4, 2, 2),
  31 :  (3,  5, 2, 2),
  24 :  (3,  6, 2, 1),
  26 :  (3,  7, 2, 1),
  17 :  (2, -1,-1,-1),
  18 :  (2, -1,-1,-1),
  19 :  (2,  0, 2, 0),
  21 :  (2,  1, 2, 0),
  32 :  (2,  2, 2, 3),
  34 :  (2,  3, 2, 3),
  27 :  (2, -1,-1,-1),
  28 :  (2,  4, 2, 2),
  30 :  (2,  5, 2, 2),
  23 :  (2,  6, 2, 1),
  25 :  (2,  7, 2, 1),
  14 :  (1,  0, 1, 3),
  16 :  (1,  1, 1, 3),
  10 :  (1,  2, 1, 2),
  12 :  (1,  3, 1, 2),
  1  :  (1,  4, 1, 0),
  3  :  (1,  5, 1, 0),
  5  :  (1,  6, 1, 1),
  7  :  (1,  7, 1, 1),
  8  :  (1, -1,-1,-1),
  13 :  (0,  0, 1, 3),
  15 :  (0,  1, 1, 3),
  9  :  (0,  2, 1, 2),
  11 :  (0,  3, 1, 2),
  0  :  (0,  4, 1, 0),
  2  :  (0,  5, 1, 0),
  4  :  (0,  6, 1, 1),
  6  :  (0,  7, 1, 1),
}

# special case for A5 board with uneven number of rows
channel_number_to_channel_location_A5 = {
  46 :  (4,  0, 3, 1),
  45 :  (4,  1, 3, 1),
  49 :  (4,  2, 3, 0),
  50 :  (4,  3, 3, 0),
  37 :  (4,  4, 3, 3),
  36 :  (4,  5, 3, 3),
  40 :  (4,  6, 3, 2),
  41 :  (4,  7, 3, 2),
  20 :  (3,  0, 2, 0),
  22 :  (3,  1, 2, 0),
  33 :  (3,  2, 2, 3),
  35 :  (3,  3, 2, 3),
  29 :  (3,  4, 2, 2),
  31 :  (3,  5, 2, 2),
  24 :  (3,  6, 2, 1),
  26 :  (3,  7, 2, 1),
  17 :  (2, -1,-1,-1),
  18 :  (2, -1,-1,-1),
  19 :  (2,  0, 2, 0),
  21 :  (2,  1, 2, 0),
  32 :  (2,  2, 2, 3),
  34 :  (2,  3, 2, 3),
  27 :  (2, -1,-1,-1),
  28 :  (2,  4, 2, 2),
  30 :  (2,  5, 2, 2),
  23 :  (2,  6, 2, 1),
  25 :  (2,  7, 2, 1),
  14 :  (1,  0, 1, 3),
  16 :  (1,  1, 1, 3),
  10 :  (1,  2, 1, 2),
  12 :  (1,  3, 1, 2),
  1  :  (1,  4, 1, 0),
  3  :  (1,  5, 1, 0),
  5  :  (1,  6, 1, 1),
  7  :  (1,  7, 1, 1),
  8  :  (1, -1,-1,-1),
  13 :  (0,  0, 1, 3),
  15 :  (0,  1, 1, 3),
  9  :  (0,  2, 1, 2),
  11 :  (0,  3, 1, 2),
  0  :  (0,  4, 1, 0),
  2  :  (0,  5, 1, 0),
  4  :  (0,  6, 1, 1),
  6  :  (0,  7, 1, 1),
}

# HD tileboards with 12 tiles / ring (12 iphi)
channel_number_to_channel_location_12phi = {
  # channel : (ring order, iphi, "Trig.-Out", "TC")
  #HGCROCC2
  #HalfROC 0
  165  :  (11,  0,  4, 3),
  166  :  (11,  1,  4, 3),
  167  :  (11,  2,  4, 3),
  158  :  (11,  3,  4, 2),
  156  :  (11,  4,  4, 2),
  157  :  (11,  5,  4, 2),
  147  :  (11,  6,  4, 1),
  148  :  (11,  7,  4, 1),
  149  :  (11,  8,  4, 1),
  140  :  (11,  9,  4, 0),
  138  :  (11,  10, 4, 0),
  139  :  (11,  11, 4, 0),

  164  :  (10,  0,  4, 3),
  163  :  (10,  1,  4, 3),
  168  :  (10,  2,  4, 3),
  159  :  (10,  3,  4, 2),
  155  :  (10,  4,  4, 2),
  154  :  (10,  5,  4, 2),
  146  :  (10,  6,  4, 1),
  145  :  (10,  7,  4, 1),
  150  :  (10,  8,  4, 1),
  141  :  (10,  9,  4, 0),
  137  :  (10,  10, 4, 0),
  136  :  (10,  11, 4, 0),

  171  :  (9,  0,  4, 3),
  170  :  (9,  1,  4, 3),
  169  :  (9,  2,  4, 3),
  160  :  (9,  3,  4, 2),
  161  :  (9,  4,  4, 2),
  162  :  (9,  5,  4, 2),
  153  :  (9,  6,  4, 1),
  152  :  (9,  7,  4, 1),
  151  :  (9,  8,  4, 1),
  142  :  (9,  9,  4, 0),
  143  :  (9,  10, 4, 0),
  144  :  (9,  11, 4, 0),

  #HalfROC 1
  133  :  (8,  0,  3, 3),
  134  :  (8,  1,  3, 3),
  135  :  (8,  2,  3, 3),
  126  :  (8,  3,  3, 2),
  125  :  (8,  4,  3, 2),
  124  :  (8,  5,  3, 2),
  115  :  (8,  6,  3, 1),
  116  :  (8,  7,  3, 1),
  117  :  (8,  8,  3, 1),
  108  :  (8,  9,  3, 0),
  107  :  (8,  10, 3, 0),
  106  :  (8,  11, 3, 0),

  129  :  (7,  0,  3, 3),
  130  :  (7,  1,  3, 3),
  132  :  (7,  2,  3, 3),
  123  :  (7,  3,  3, 2),
  120  :  (7,  4,  3, 2),
  121  :  (7,  5,  3, 2),
  111  :  (7,  6,  3, 1),
  112  :  (7,  7,  3, 1),
  114  :  (7,  8,  3, 1),
  105  :  (7,  9,  3, 0),
  101  :  (7,  10, 3, 0),
  103  :  (7,  11, 3, 0),

  128  :  (6,  0,  3, 3),
  127  :  (6,  1,  3, 3),
  131  :  (6,  2,  3, 3),
  122  :  (6,  3,  3, 2),
  119  :  (6,  4,  3, 2),
  118  :  (6,  5,  3, 2),
  110  :  (6,  6,  3, 1),
  109  :  (6,  7,  3, 1),
  113  :  (6,  8,  3, 1),
  104  :  (6,  9,  3, 0),
  100  :  (6,  10, 3, 0),
  102  :  (6,  11, 3, 0),

# HGCROCC1
  65  :  (5,  0,  2, 3),
  66  :  (5,  1,  2, 3),
  67  :  (5,  2,  2, 3),
  58  :  (5,  3,  2, 2),
  56  :  (5,  4,  2, 2),
  57  :  (5,  5,  2, 2),
  47  :  (5,  6,  2, 1),
  48  :  (5,  7,  2, 1),
  49  :  (5,  8,  2, 1),
  40  :  (5,  9,  2, 0),
  38  :  (5,  10, 2, 0),
  39  :  (5,  11, 2, 0),

  64  :  (4,  0,  2, 3),
  63  :  (4,  1,  2, 3),
  68  :  (4,  2,  2, 3),
  59  :  (4,  3,  2, 2),
  55  :  (4,  4,  2, 2),
  54  :  (4,  5,  2, 2),
  46  :  (4,  6,  2, 1),
  45  :  (4,  7,  2, 1),
  50  :  (4,  8,  2, 1),
  41  :  (4,  9,  2, 0),
  37  :  (4,  10, 2, 0),
  36  :  (4,  11, 2, 0),

  71  :  (3,  0,  2, 3),
  70  :  (3,  1,  2, 3),
  69  :  (3,  2,  2, 3),
  60  :  (3,  3,  2, 2),
  61  :  (3,  4,  2, 2),
  62  :  (3,  5,  2, 2),
  53  :  (3,  6,  2, 1),
  52  :  (3,  7,  2, 1),
  51  :  (3,  8,  2, 1),
  42  :  (3,  9,  2, 0),
  43  :  (3,  10, 2, 0),
  44  :  (3,  11, 2, 0),

  33  :  (2,  0,  1, 3),
  34  :  (2,  1,  1, 3),
  35  :  (2,  2,  1, 3),
  26  :  (2,  3,  1, 2),
  25  :  (2,  4,  1, 2),
  24  :  (2,  5,  1, 2),
  15  :  (2,  6,  1, 1),
  16  :  (2,  7,  1, 1),
  17  :  (2,  8,  1, 1),
  8   :  (2,  9,  1, 0),
  7   :  (2,  10, 1, 0),
  6   :  (2,  11, 1, 0),

  29  :  (1,  0,  1, 3),
  30  :  (1,  1,  1, 3),
  32  :  (1,  2,  1, 3),
  23  :  (1,  3,  1, 2),
  20  :  (1,  4,  1, 2),
  21  :  (1,  5,  1, 2),
  11  :  (1,  6,  1, 1),
  12  :  (1,  7,  1, 1),
  14  :  (1,  8,  1, 1),
  5   :  (1,  9,  1, 0),
  1   :  (1,  10, 1, 0),
  3   :  (1,  11, 1, 0),

  28  :  (0,  0,  1, 3),
  27  :  (0,  1,  1, 3),
  31  :  (0,  2,  1, 3),
  22  :  (0,  3,  1, 2),
  19  :  (0,  4,  1, 2),
  18  :  (0,  5,  1, 2),
  10  :  (0,  6,  1, 1),
  9   :  (0,  7,  1, 1),
  13  :  (0,  8,  1, 1),
  4   :  (0,  9,  1, 0),
  0   :  (0,  10, 1, 0),
  2   :  (0,  11, 1, 0),
}

"""
Number of rows, iphi and maximum ring number for each module type.
"""
module_types = {
#  type          #nrows   #phi    maxring
  "TM-A5" :   (5,       8,      5   ),
  "TM-A6" :   (6,       8,      5   ),
  "TM-B11B12" : (12,      8,      17  ), 
  "TM-B12" :    (12,      8,      17  ), 
  "TM-C5" :     (5,       8,      17  ),
  "TM-D8" :     (8,       8,      25  ),
  "TM-E8" :     (8,       8,      33  ),
  "TM-G3" :     (3,       8,      36  ),
  "TM-G4" :     (4,       8,      37  ),
  "TM-G5" :     (5,       8,      38  ),
  "TM-G6" :     (6,       8,      39  ),
  "TM-G7" :     (7,       8,      40  ),
  "TM-G8" :     (8,       8,      41  ),
  "TM-J8" :     (8,       8,      25  ),
  "TM-J12" :    (12,      12,     11  ), #h11
  "TM-K4" :     (4,       8,      29  ),
  "TM-K5" :     (5,       12,     16  ), #h16
  "TM-K6" :     (6,       12,     17  ),
  "TM-K7" :     (7,       8,      32  ),
  "TM-K8" :     (8,       12,     19  ), #h19
  "TM-K10" :    (10,      12,     21  ), #h21
  "TM-K11" :    (11,      12,     22  ), #h22
  "TM-K12" :    (12,      12,     23  )  #h23
}

""" Definition of which tileboards are considered HD."""
HDtypes = ("TM-K6", "TM-K8", "TM-K11", "TM-K12", "TM-J12")

"""
ECON-D input ordering of tileboard halfROCs for each layer.
For each layer the tileboard typecode and the halfROC number is given.
NOTE: The odering of HD tileboards could come to change in the future, and are currently more of a placeholder.
"""
layer_ordering = {
  34 : (("TM-K6",1), ("TM-J12", 1), ("TM-J12",3), ("TM-J12",2), ("TM-J12",4), ("TM-K6",2)),
  35 : (("TM-K8",1), ("TM-K8",3), ("TM-J12", 1), ("TM-J12",3), ("TM-J12",2), ("TM-J12",4), ("TM-K8",2)),
  36 : (("TM-K11",1), ("TM-K11",3), ("TM-J12", 1), ("TM-J12",3), ("TM-J12",2), ("TM-J12",4), ("TM-K11",2), ("TM-K11",4)),
  37 : (("TM-K12",1), ("TM-K12",3), ("TM-J12", 1), ("TM-J12",3), ("TM-J12",2), ("TM-J12",4), ("TM-K12",2), ("TM-K12",4)),
  38 : (("TM-D8", 1), ("TM-E8",1), ("TM-C5",1), ("TM-G3",1), ("TM-G3",2), ("TM-C5",2), ("TM-D8",2), ("TM-E8",2)),
  39 : (("TM-D8", 1), ("TM-E8",1), ("TM-C5",1), ("TM-G5",1), ("TM-G5",2), ("TM-C5",2), ("TM-D8",2), ("TM-E8",2)),
  40 : (("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G7",1), ("TM-G7",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2)),
  41 : (("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G8",1), ("TM-G8",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2)),
  42 : (("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G8",1), ("TM-G8",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2)),
  43 : (("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G8",1), ("TM-G8",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2)),
  44 : (("TM-A5",1), ("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G8",1), ("TM-G8",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2), ("TM-A5",2)),
  44 : (("TM-A6",1), ("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G8",1), ("TM-G8",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2), ("TM-A6",2)),
  45 : (("TM-A6",1), ("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G8",1), ("TM-G8",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2), ("TM-A6",2)),
  46 : (("TM-A6",1), ("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G5",1), ("TM-G5",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2), ("TM-A6",2)),
  47 : (("TM-A6",1), ("TM-D8",1), ("TM-E8",1), ("TM-B11B12",1), ("TM-B11B12",3), ("TM-G5",1), ("TM-G5",2), ("TM-B11B12",2), ("TM-D8",2), ("TM-E8",2), ("TM-A6",2)),
}

if __name__ == '__main__':

  filename = 'channels_sipmontile.hgcal.txt'

  header = ("Typecode", "ROC", "HalfROC", "Seq", "ROCpin", "TrLink", "TrCell", "iring", "iphi", "t", "trace", "localROC", "localHalfROC",)

  with open(filename,'w') as f:
    for h in header:
      f.write(h)
      f.write(" ")
    f.write("\n")


    for layer, modules in layer_ordering.items():
      
      for sector in (1, 2, 3):
        # global ROC and HalfROC counters
        globalROC = 0
        globalHalfROC = 0
        for module in modules:
          modtype = module[0]
          halfROC_ = module[1]
          localROC = 1 if halfROC_ < 3 else 2
          localHalfROC = halfROC_ if halfROC_ < 3 else halfROC_ - 2

          mod_n_rings = module_types[modtype][0]
          n_phis = module_types[modtype][1]
          mod_ring_max = module_types[modtype][2]

          hd_tiles = n_phis == 12
          # Get max ring number and number of rings for the module
          ring_max, n_rings = get_ring_max_and_n_rings(mod_n_rings, mod_ring_max, localROC, localHalfROC, hd_tiles)

          ring_min = 0
          if (halfROC_ == 2):
            ring_min += 4
          if (halfROC_ == 3):
            ring_min += 8
          # Get channel map
          if not hd_tiles:
            channel_number_to_channel_location = channel_number_to_channel_location_8phi
            if modtype == "TM-A5":
              channel_number_to_channel_location = channel_number_to_channel_location_A5
          if hd_tiles:
            channel_number_to_channel_location = channel_number_to_channel_location_12phi

          # sort channel_number_to_channel_location after channel number
          channel_number_to_channel_location = dict(sorted(channel_number_to_channel_location.items()))

          for channel,locations in channel_number_to_channel_location.items():
            local_ring = locations[0]
            base_iphi = locations[1]
            trigout = locations[2]
            trigch = locations[3]

            # channel is outside of the current module's ring range
            if local_ring < ring_min:
              continue
            if local_ring-ring_min >= n_rings:
              continue
            
            ring = get_global_ring_number(local_ring-ring_min, ring_max)
            iphi = get_iphi_number(base_iphi, sector, hd_tiles)
    
            isHD = False if not hd_tiles else True

            seq = get_sequence_for_channel(channel, isHD)

            if(trigch<0):
              ring_str="-1"
              t=-1
            else:
              ring_str = str(ring)
              t=1

            # define typecode
            density = "H" if hd_tiles else "L"
            typecode = "T"+str(density)+"-"+"L"+str(layer)+"S"+str(sector)

            # write to file
            f.write(f'{typecode} {globalROC} {globalHalfROC} {seq} {channel} {trigout} {trigch} {ring_str} {iphi} {t} {0} {localROC} {localHalfROC}\n')

          # increment global counters
          if globalHalfROC >= 1:
            globalHalfROC = 0
            globalROC += 1
          else:
            globalHalfROC += 1
    
print(f"File {filename} created with {len(layer_ordering)} layers and {sum(len(modules) for modules in layer_ordering.values())} halfROCs.")
