#pragma once
struct cluster_property{
                      
     bool low_pt_trk_cluster;
     bool high_pt_trk_cluster;
     float barycenter;
     int size;
     int firstStrip;
     int endStrip;
     int charge;
     int trk_algo;

     cluster_property() : low_pt_trk_cluster(0), high_pt_trk_cluster(0), barycenter(-1),
                          size(-1), firstStrip(-1), endStrip(-1), charge(-1), trk_algo(-1)
                       {};
     cluster_property(bool in_low_pt_trk_cluster, bool in_high_pt_trk_cluster,
                      float in_barycenter, int in_size, int in_firstStrip, int in_endStrip,
                      int in_charge, int in_trk_algo):
                     low_pt_trk_cluster(in_low_pt_trk_cluster),
                     high_pt_trk_cluster(in_high_pt_trk_cluster),
                     barycenter(in_barycenter),
                     size(in_size), firstStrip(in_firstStrip), endStrip(in_endStrip),
                     charge(in_charge),
                     trk_algo(in_trk_algo)
                     {};

};
