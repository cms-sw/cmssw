#ifndef L1Trigger_DTTriggerPhase2_vhdl_h
#define L1Trigger_DTTriggerPhase2_vhdl_h

#include <cstdint>
#include <vector>
#include <cmath>

// "à la vhdl" functions
std::vector<int> vhdl_slice(std::vector<int> v, int upper, int lower);
int vhdl_unsigned_to_int(std::vector<int> v);
int vhdl_signed_to_int(std::vector<int> v);
void vhdl_int_to_unsigned(int value, std::vector<int> &v);
void vhdl_int_to_signed(int value, std::vector<int> &v);
void vhdl_resize_unsigned(std::vector<int> &v, int new_size);
void vhdl_resize_signed(std::vector<int> &v, int new_size);
bool vhdl_resize_signed_ok(std::vector<int> v, int new_size);
bool vhdl_resize_unsigned_ok(std::vector<int> v, int new_size);

#endif
