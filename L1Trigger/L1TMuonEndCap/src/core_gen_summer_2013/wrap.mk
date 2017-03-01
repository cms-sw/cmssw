# This Makefile was automatically generated
# by VPPC from a Verilog HDL project.
# VPPC web-page: http://www.phys.ufl.edu/~madorsky/vppc/

# Author    : madorsky
# Timestamp : Tue May 18 09:11:53 2010

debug=
gpp=g++-4
opt=-O0
sim_lib=../sim_lib_src

all: sp_wrap_tf

sp_wrap_tf: best_tracks.o coord_delay.o deltas.o best_delta.o deltas_sector.o extend_sector.o extender.o find_segment.o \
       match_ph_segments.o ph_pattern.o ph_pattern_sector.o prim_conv.o prim_conv11.o prim_conv_sector.o sort_sector.o \
       zone_best.o zone_best3.o sp.o  zones.o vppc_sim_lib.o sp_wrap.o sp_wrap_tf.o
	$(gpp) -Wall $(debug) $(opt) -o $@ $^

sp_wrap.o: sp_wrap.cpp sp_wrap.h sp.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

sp_wrap_tf.o: sp_wrap_tf.cpp sp_wrap.h sp.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

best_tracks.o : best_tracks.cpp best_tracks.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

coord_delay.o : coord_delay.cpp coord_delay.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

deltas.o : deltas.cpp deltas.h best_delta.h best_delta.h best_delta.h best_delta.h best_delta.h best_delta.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

best_delta.o : best_delta.cpp best_delta.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

deltas_sector.o : deltas_sector.cpp deltas_sector.h deltas.h deltas.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

extend_sector.o : extend_sector.cpp extend_sector.h extender.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

extender.o : extender.cpp extender.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

find_segment.o : find_segment.cpp find_segment.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

match_ph_segments.o : match_ph_segments.cpp match_ph_segments.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h find_segment.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

ph_pattern.o : ph_pattern.cpp ph_pattern.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

ph_pattern_sector.o : ph_pattern_sector.cpp ph_pattern_sector.h ph_pattern.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

prim_conv.o : prim_conv.cpp prim_conv.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

prim_conv11.o : prim_conv11.cpp prim_conv11.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

prim_conv_sector.o : prim_conv_sector.cpp prim_conv_sector.h prim_conv11.h prim_conv.h prim_conv.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

sort_sector.o : sort_sector.cpp sort_sector.h zone_best3.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

zone_best.o : zone_best.cpp zone_best.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

zone_best3.o : zone_best3.cpp zone_best3.h zone_best.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

sp.o : sp.cpp sp.h prim_conv_sector.h zones.h extend_sector.h ph_pattern_sector.h sort_sector.h coord_delay.h match_ph_segments.h deltas_sector.h best_tracks.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

zones.o : zones.cpp zones.h $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<

vppc_sim_lib.o: $(sim_lib)/vppc_sim_lib.cpp $(sim_lib)/vppc_sim_lib.h 
	$(gpp) -Wall $(debug) $(opt) -I$(sim_lib) -o $@ -c $<








clean:
	rm -rf *.o *.exe
