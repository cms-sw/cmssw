# `ngt::AnyBuffer`

`ngt::AnyBuffer` behaves like `std::any`, with two differences: it can only be
used with trivially copyable types, and provides access to the underlying memory
buffer to allow `memcpy`'ing its content.
