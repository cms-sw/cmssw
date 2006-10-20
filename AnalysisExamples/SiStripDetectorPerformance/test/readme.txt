The config file selects 3.8 Tesla runs. You can find, commented, a subset of run with different magnetic field configuration.

In the output rootuple you will find a set of variables related to the track, trigger and clusters.

Track-related variables:
Ntk
p_tk
pt_tk
eta_tk
phi_tk
nhits_tk 

Trigger-related variables (from LTC digi collection):
DTtrig
NODTtrig
CSCtrig
Othertrig

Cluster-related variables only for hits used for track reconstruction:
Nclu (cluster number)
Subid (==3 TIB, ==5 TOB, == 6 TEC)
Layer (1 or 2 for internal or external)
Clu_rawid (detector raw ID)
Clu_ch (cluster charge)
Clu_ang (angle between track and detector respect to the x local coordinate)
Clu_size (cluster size)
Clu_bar (cluster barycenter)
Clu_1strip (position of the first strip of the cluster)

Cluster-related variables only for hits used for track reconstruction:
same as above but with _all suffix



livio.fano@cern.ch

