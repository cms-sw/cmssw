// The purpose of tis is to run in totalview while the catch in MessageSender
// destructor is **disabled** (commented out) to verify that we have cured
// the use-of-threads-after-main headache.

int main() { return 0; }
