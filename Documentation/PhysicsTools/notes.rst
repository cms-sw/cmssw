=====================================
Development Guidelines and Principles
=====================================

.. contents:: Contents

Declaration of Principles and Guidelines
========================================

User interactions are most important.

The number of rules required to perform an activity 
properly should be minimized.

Complimentary operations should be used similarly.

Generate objects that match the users "mental model" 
for solving the problem. 

Common boilerplate code within a user's program should be minimized.
It represents something that will be left out or become unchangable,
or will evolve into something poor and wrong.

Generate objects with a single purpose.  Objects created by
cleanly separating concerns during analysis are more readily
usuable in a broader range of contexts
and can also be more easily used to build larger components with more
complex behavior.

Requirements
============

First get the simple scheduled reconstruction working properly.
Do not create interfaces that make a demand driven system impossible.

Demand driven capabilities can be introduced after the first 
scheduled system is available.

Common job configuration
------------------------

Groups of analysis users will likely create a common job
configuration capable of reconstituting any `EDProducts` of
interest within any `Event`,
in any sample for which they are interested.

The users will not want to modify this configuration with regard to
differences in available `EDProucts` from run to run of their jobs.

The users will not want to experience any noticable overhead
associated with the extra code that may be present in their
application to accomodate any `EDProducts` that they may
encounter in their event samples (change from run to run or
event to event).

Question: In a system that has an `Event` that does not distinguish
between "reconstitution" and "creation", is their a penalty
to be paid for having more `EDProduct` code available than is
really used by the `Analysis` modules? Does each event of
each run need to contain an `EDProduct` index, which is used 
to populate "reconstitutors" - one for each product that the
executable knows about? If no, then how does one express which
"reconstitutors" to populate?


Application of Principles and Guidelines
========================================


How do users access data products for a given collision?
--------------------------------------------------------


Solution One
++++++++++++

Data products are accessed through a `Getter`.
`Getters` maintain an association with the current `Event`.

`Getters` have appeared under two solutions:

 1.  One variety of `Getter`  uses the `Event` interface internally:
     it is passed an `Event`, and reaches into the `Event` to find
     the datum it (the `Getter`) is to return.

 2. The other variety of `Getter` is wired directly to the output of
    the `EDProducer` (or `Provider`) that provides the `EDProduct`
    the `Getter` is intended to return.
    Multiple `Getters` are synchronized
    so that they obtain `EDProducts` from the same collision.

User code interacts directly with the `Getter`.
It is not necessary for the user to see the `Event`,
or even for the `Event` to exist.

Solution Two
++++++++++++

Data products are accessed through the `Event`.

The "get" method of the `Event` is used to acquire `EDProducts`.
This method is parameterized 
on the specific type of `EDProduct` to be retrieved.

A `Selector` is used to specify *which* `EDProduct`
(or `EDProducts`)
of that type  is (or are)
to be returned.

Solution Three
++++++++++++++

Data products are accessed through the `Event`.

The "get" method of the `Event` is used to acquire `EDProducts`.
This method is parameterized
on the specific type of `EDProduct` to be retrieved.

A `Label` is a simple human readable string
(or group of strings)
that uniquely identifies an `EDProduct` of a certain type.
A `Label` is used to specify *which* `EDProduct` of that type
is to be returned.
`Labels` are always meant to be unique.

Unlike a `Selector`, 
a `Label` does not itself determine what `EDProduct` matches it.
Something outside the `Label` and `EDProduct`
performs the comparison.

How do we put things into the event?
------------------------------------

Solution One
++++++++++++

Data products are allocated
and inserted
through the `Event`.

The `Event` contains a "put" method.
This method is parameterized on the type of `EDProduct`
and is used to allocate an object
that will automatically be placed permanently into the `Event`
when the `EDProducer` successfully completes its task.

`EProducers` that make more than one `EDProduct`
will invoke "put" for each of the products they generate.
All the products will be successfully added to the `Event`,
or none will be added.

Provanance information is added automatically
by the framework
during the final commit.
The user is not responsible
for handling this information.

Solution Two
++++++++++++

Data products are allocated dynamically
and returned by the "produce" method of the `EDProducer`.
Ownership of the created products is passed during the return.

The framework handles inserting the created products
into the `Event`.
Provanance information is added automatically
by the framework.

How do we access conditions data associated with an event?
----------------------------------------------------------

The framework will associate each event with a
complete set of calibration and alignment objects
(and any other conditions data)
that are correct for that event.
The user does not supply arguments
to specify the proper conditions objects
associated with the event being processed.

We anticipate that different categories of conditions data
(conditions subsystems)
will have different interfaces to retrieve the data.
Every subsystem will label objects with a numeric ID.

Should `Producer` developers register things required and things produced?
--------------------------------------------------------------------------

Registering things required is most useful
in validating an execution schedule.

Registering things produced is necessary for a demand driven system
and is useful for validation during configuration of a scheduled system.

Should things produced or required information be available at compile-time?
----------------------------------------------------------------------------

What kind of information should be available at compile time
concerning things produced or required?

Open Questions
==============

Many problems are arising
when we consider allowing multiple objects of the same type
to be produced by a single `EDProducer`.
The source of the problem
is the fact that the produced objects
are distinguishable neither by
type
nor by
provenance (which describes the configuration
of the producer
and the "context" in which the producer was run).
Thus,
to distinguish between multiple objects of the same type
produced by a single `EDProducer`,
one is forced to look at
the data of the `EDProduct` itself.

The problems include:

1. Schedule validation becomes difficult.
   It seems to require creation of prototype instances
   of the `EDProducts` at configuration time,
   so that the relevant data can be matched
   (by a `Selector` that knows about that specific
   `EDProduct`).

2. The need to create these prototype objects
   limits our flexibility in having `EDProducers`
   announce what they make.

3. It requires that we support `Selectors` that look
   at `EDProducts` (and concrete subclasses),
   not just `Provenances`.
   This puts a greater demand on the authors
   of `EDProducers` and `EDProducts`
   to create the relevant `Selectors`.
   Previous experience leads us to believe
   it will be difficult to assure all
   `EDProduct` designers
   will produce the appropriate `Selector` classes.

We propose that `Selectors`
passed to the `Event` should only operate on
`Provenances`.

When combined with the requirement
that `EDProducers`
use only the `Event::get` function
which returns a single `EDProduct`,
there is a drawback to this choice.
It means that `EDProducers`
can not make use of the output of 
other `EDProducers` which make multiple instances
of the same type.
One way around this
is for the `Provenance` to carry
a user-supplied bit of data
which can then be used by the `Selector`
to identify a single matching `EDProduct`.
