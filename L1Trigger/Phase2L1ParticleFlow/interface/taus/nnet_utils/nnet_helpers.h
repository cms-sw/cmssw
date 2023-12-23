#ifndef NNET_HELPERS_H
#define NNET_HELPERS_H


#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace nnet {

constexpr int ceillog2(int x) { return (x <= 2) ? 1 : 1 + ceillog2((x + 1) / 2); }

} // namespace nnet

#endif
