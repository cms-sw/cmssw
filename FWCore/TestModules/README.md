# General-purpose modules for testing purposes

This package contains modules that are used in framework tests, but
are generic-enough to be usable outside of the framework as well.
Their interfaces are intended to be relatively stable.

## `edmtest::GlobalIntProducer`, `edmtest::GlobalFloatProducer`, `edmtest::GlobalStringProducer`, `edmtest::GlobalVectorProducer`

These modules can be used to produce into the event plain C++ data types based
on configurable values: a single `int`, a single `float`, a single `std::string`,
a `vector<double>`.


## `edmtest::GlobalIntAnalyzer`, `edmtest::GlobalFloatAnalyzer`, `edmtest::GlobalStringAnalyzer`, `edmtest::GlobalVectorAnalyzer`

These modules can be used to read form the event plain C++ data types, and
compare them with configurable expected values: a single `int`, a single `float`,
a single `std::string`, a `vector<double>`.


## `edmtest::StreamIDFilter`

This module can be used to reject all events in specific streams.


## `edmtest::EventIDProducer`

This module reads the `EventID` from the current event and copies it as a data
product into the `Event`.


## `edmtest::EventIDValidator`

This module reads the `EventID` from the current event and compares it to a data
product read from the `Event`.

Together `edmtest::EventIDProducer` and `edmtest::EventIDValidator` can be used
to validate that an object produced in a given event is being read back in the
same event.


## `RunLumiEventAnalyzer`

This module can be used to enforce the order of Runs, LuminosityBlocks, and Events the module sees. Note that this module is not useful in a job that uses multiple streams, because the exact order of Runs/LuminosityBlocks/Events the module sees in such jobs is not deterministic.


## `SecondaryProducer`

This module can be used to test `cms.SecSource`.