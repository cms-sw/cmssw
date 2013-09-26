/*
 * LegendItem.cc
 *
 *  Created on: Sep 24, 2012
 *      Author: aspataru
 */

#include "../interface/LegendItem.h"

using namespace jsoncollector;
using std::string;

LegendItem::LegendItem(string name, string operation) :
	name_(name), operation_(operation) {
}

LegendItem::~LegendItem() {
}
