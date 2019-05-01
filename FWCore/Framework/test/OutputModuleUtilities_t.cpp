namespace edm {
  namespace test {
    // Implemented in OutputModule.cc
    void run_all_output_module_tests();
  }  // namespace test
}  // namespace edm

int main() { edm::test::run_all_output_module_tests(); }
