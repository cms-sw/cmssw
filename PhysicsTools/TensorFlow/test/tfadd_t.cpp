#include "PhysicsTools/TensorFlow/test/test_graph_tfadd/header.h"
#include <cassert>
#include <iostream>

#define EXPECT_EQ(x,y) assert(x==y)
#define	EXPECT_TRUE(x) assert(x)

int main() {
 using AddComp=test_graph_tfadd;
 typedef int int32;
 {

  std::cout << "testing tf add " << std::endl;
  AddComp add;
  EXPECT_EQ(add.arg0_data(), add.args()[0]);
  EXPECT_EQ(add.arg1_data(), add.args()[1]);

  add.arg0() = 1;
  add.arg1() = 2;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 3);
  EXPECT_EQ(add.result0_data()[0], 3);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  add.arg0_data()[0] = 123;
  add.arg1_data()[0] = 456;
  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 579);
  EXPECT_EQ(add.result0_data()[0], 579);
  EXPECT_EQ(add.result0_data(), add.results()[0]);

  const AddComp& add_const = add;
  EXPECT_EQ(add_const.error_msg(), "");
  EXPECT_EQ(add_const.arg0(), 123);
  EXPECT_EQ(add_const.arg0_data()[0], 123);
  EXPECT_EQ(add_const.arg0_data(), add.args()[0]);
  EXPECT_EQ(add_const.arg1(), 456);
  EXPECT_EQ(add_const.arg1_data()[0], 456);
  EXPECT_EQ(add_const.arg1_data(), add.args()[1]);
  EXPECT_EQ(add_const.result0(), 579);
  EXPECT_EQ(add_const.result0_data()[0], 579);
  EXPECT_EQ(add_const.result0_data(), add_const.results()[0]);
 }

  // Run tests that use set_argN_data separately, to avoid accidentally re-using
  // non-existent buffers.
 {
   std::cout << "testing tf add no input buffer" << std::endl;
   AddComp add(AddComp::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);

  int32 arg_x = 10;
  int32 arg_y = 32;
  add.set_arg0_data(&arg_x);
  add.set_arg1_data(&arg_y);
  EXPECT_EQ(add.arg0_data(), add.args()[0]);
  EXPECT_EQ(add.arg1_data(), add.args()[1]);

  EXPECT_TRUE(add.Run());
  EXPECT_EQ(add.error_msg(), "");
  EXPECT_EQ(add.result0(), 42);
  EXPECT_EQ(add.result0_data()[0], 42);
  EXPECT_EQ(add.result0_data(), add.results()[0]);
 }

}
