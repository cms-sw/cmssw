#include "L1Trigger/DTTriggerPhase2/interface/vhdl_functions.h"

// "a la vhdl" functions
std::vector<int> vhdl_slice(std::vector<int> v, int upper, int lower) {
  int final_value = lower;
  if (final_value < 0)
    final_value = 0;

  std::vector<int> v1;
  for (int i = final_value; i <= upper; i++) {
    v1.push_back(v[i]);
  }
  return v1;
}

int vhdl_unsigned_to_int(std::vector<int> v) {
  int res = 0;

  for (size_t i = 0; i < v.size(); i++) {
    res = res + v[i] * std::pow(2, i);
  }
  return res;
}

int vhdl_signed_to_int(std::vector<int> v) {
  if (v[v.size() - 1] == 0)
    return vhdl_unsigned_to_int(v);
  else
    return -(std::pow(2, v.size()) - vhdl_unsigned_to_int(v));
}

void vhdl_int_to_unsigned(int value, std::vector<int> &v) {
  if (value == 0) {
    v.push_back(0);
  } else if (value != 1) {
    v.push_back(value % 2);
    vhdl_int_to_unsigned(value / 2, v);
  } else {
    v.push_back(1);
  }
  return;
}

void vhdl_int_to_signed(int value, std::vector<int> &v) {
  if (value < 0) {
    int val = 1;
    int size = 1;
    while (val < -value) {
      val *= 2;
      size += 1;
    }
    vhdl_int_to_unsigned(val + value, v);
    for (int i = v.size(); i < size - 1; i++) {
      v.push_back(0);
    }
    v.push_back(1);
  } else {
    vhdl_int_to_unsigned(value, v);
    v.push_back(0);
  }
  return;
}

void vhdl_resize_unsigned(std::vector<int> &v, int new_size) {
  for (int i = v.size(); i < new_size; i++) {
    v.push_back(0);
  }
}

void vhdl_resize_signed(std::vector<int> &v, int new_size) {
  int elem = 0;
  if (v[v.size() - 1] == 1)
    elem = 1;
  for (int i = v.size(); i < new_size; i++) {
    v.push_back(elem);
  }
}

bool vhdl_resize_signed_ok(std::vector<int> v, int new_size) {
  for (size_t i = v.size() - 1 - 1; i >= v.size() - 1 - (v.size() - new_size); i--) {
    if (v[i] != v[v.size() - 1])
      return false;
  }
  return true;
};

bool vhdl_resize_unsigned_ok(std::vector<int> v, int new_size) {
  for (size_t i = v.size() - 1; i >= v.size() - 1 + 1 - (v.size() - new_size); i--) {
    if (v[i] != 0)
      return false;
  }
  return true;
};
