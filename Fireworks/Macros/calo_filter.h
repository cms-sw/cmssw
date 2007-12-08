#ifndef calo_filter_h
#define calo_filter_h

class TEveElement;

enum {
  do_nothing    = 0,
  do_hide       = 1,
  do_remove     = 2
};

void node_filter(TEveElement * node, int simplify = do_hide);
void apply_filter(TEveElement * node, int simplify = do_hide);

#endif // calo_filter_h
