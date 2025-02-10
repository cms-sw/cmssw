# General-purpose modules for testing purposes

This package contains modules that are used in framework tests, but
are generic-enough to be usable outside of the framework as well.
Their interfaces are intended to be relatively stable.


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
