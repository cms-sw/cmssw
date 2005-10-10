===============
Framework tasks
===============

Overview
========

For each task, we identify the following products
(some tasks may not have code deliverables).
Each product should be produced in order;
we should review after each product
before moving on to the next.

D. Design products: documentation, in the RoadMap, describing
   the how and why of the design, and prototype interface
   consisting of public headers in the repository.

I. Implementation: working code, in the repository, with
   functioning unit tests. It may be necessary to modify
   the prototype interfaces at this time.

C. Integration of the product into the remainder of the core
   software. More modification of interface may be needed.
   Example use in a framework program context (when relevant)
   should be included.

R. Review with others.

Stages are:
  D=design,
  I=implementation and unit testing,
  C=complete integration of task
  R=review of the task
  A=D, I, and C

We expect that most tasks will have incremental releases, each
including more functionality.  Each step or item within a 
task has a release phase number assigned to it.
This phase number is relative to the task.
Earlier phases indicate higher priority.

Until we invent or find a better system, each part (or step)
of a task is tagged with:
 <stage>(phase, required_input_in_terms_of_the_task_step_list)
 
If dutput products are included anywhere, they will be stated 
as object names i.e. single strings.

Notes
-----
1) Design of a deliverable depends only on design of inputs
2) Implementation (up to unit testing) depends only on design
   and public headers of inputs.
3) Complete integration of a deliverable depends can complete
   integration of its input deliverables

Graph production
----------------
Given a task number and phase within a the task, produce a 
dependency graph that only includes subtasks from the
given task <= the given phase number,
and all other tasks that is depends on.

Most of the subtask names below are too long to include in a graph.
Short names for subtasks will be identified {like this}.

Tasks
=====

We can identify the following deliverable products.

1. Module Registry {ModuleRegistry}

   *Modules* are things that are manipulated through *Workers*.
   ModuleRegistry, which is a caching layer on top of the ModuleFactory.
   This is used by schedule builder and such, and not ModuleFactory.
   The main interface should have the same signature as
   ModuleFactory::makeWorker(...).
   
   a. Interface for locating and iterating through worker instances,
      and for getting workers using parameter set and
      job information, and {repository for created worker} instances.
      <A>(1,none)
   b. Handling {version management} and validation for "plug-ins".
      Using "cvs tags", verify that all modules come from the same
      release.  Support a development mode, where untagged modules
      can be mixed in with released one, but provenance clearly
      indicates this.  This implies that the cvs tag string will 
      probably need to be compiled into the plugin library by
      the build system (PluginManager may already do this).
      <A>(2,1a)

   Input: ModuleFactory, Pset, Worker interface
   Output: ModuleRegistry design and version support,
    interface published, class definition and implementation

2. Scheduler Builder and Validator {ScheduleBuilder}

   Use the parsed path expression (within a process section) to
   create a schedule.  A schedule is basically one list of modules
   per path.

   a. {substitute sequence nodes} into path nodes (sequences
      are just aliases)
      <A>(1,12j)
   b. Verify that {prerequisites as declared} in path expressions are
      consistent as specified in the parsed file.
      <A>(1,2a)
   c. {Remove redundency} in each of the paths
      <A>(1,2a)
   d. Verify that {prerequisite EDProducts} are available before each
      module is invoked
      <A>(2,3c)
   e. Build ScheduleExecutor's {lists of workers} from path nodes
      <D>(1,12b,1a,2b,2c)
      <I>(1,12c,1a,2b,2c)
      <C>(1,12b,12c,1a,2b,2c)
   f. Allow requests of {reconfiguration} of modules in a schedule.
      <A>(4,?)

   Inputs: Node trees from PSet for paths and sequences,
           "ProcessDesc" object (this has PSet that represents the job and
           allows access to the parsed path nodes),
           PSet representing a job,
           ModuleRegistry(1b-D),
           "I make this..." objects (part of phase 3)
   Outputs: Schedule builder class

3. "I make this" and "I use this" interface. {announcing product requirements}

   Each EDProducer module must declare what it adds to the event and
   what it uses from the event.

   a. {EDProducer interface modification} (1)
   b. The {examples} need to be modified to use these features (1)
   c. Modify and add {code to manage} and make use of this information (2)

   Inputs: base classes for "modules", Provenance and Selectors.
   Outputs: list of dependent products and ways to get to them,
            list of things that are made if this is a producer

4. {Unscheduled Executor}

   Demand driven mode.

   a. Addition of a "proxy" map associated with the EventPrincipal that
      uses provenance information to locate an EDProducer when a EDProduct
      is requested from the Event. [3c]
   b. Locate the filters and analyzers (ordered list) to be used as the
      starting points to trigger production of necessary products. [3c]
   c. The event loop
   d. additions to configuration to support this mode (if necessary)
   e. Event/EventPrincipal modifications to support this mode

   We skip this for now.

5. Schedule Executor

   a. Get event from the event source or get handed event
      (two possible implementations) (1)
   b. Propagation of events through list of list of modules produced 
      from the Builder (the schedule). (1)
   c. Handle control flow change requested by error/message handling
      subsystem.
      Some examples of these requests are:
      - a filter terminates a path (1)
      - a producer fails to converge on a solution (2 - design first)
      - a "stop processing" request (of segment or event or other)
        was given from a Producer or Filter or other module (3)
      - a producer throws an exception (1-trivial way, 2-standard way)
   d. Handle signals (user interrupts like cntl-c, QUIT) properly (2)
   e. Configuration rules

   Trivial = fail the path, standard = configurable reactions and exception
   hierarchy

   Inputs: Design work for separation of concerns - who calls the event
    source, who has the event loop? 
    Output of schedule builder, PSet for configuration (probably from
    "ProcessDesc"), Exception heirarchy, Error/message logging facility

   Outputs: SchedulerExecutor interface and implementation.

6. Floating point exception management
 
   There are several examples of other experiments and projects that 
   do this well.  Get one of them and integrate it.

   Outputs: object for management of FPU

7. Argument parsing and processing (command line)

   Boost has a promising command line parsing library.  Attempt to use
   it here, after the requirements for such a facility are written.

   Output: object that can be used to print command line option help,
     and parse command line options and allow for overrides from
     environment variables.

8. Event Processor object

   This is the object that encapsulates all the event processing, 
   including configuration, and the event loop (which may be delegated
   to the schedule executor or another utility class). Control is
   not returned to the caller until all event processing is done
   i.e. the event loop stops

   a. Return codes from "go" method defined
   b. Allow for module reconfiguration
      0. first phase supports only untracked parameter editting
      1. may change schedule (cause rebuild due to PSet nesting)
      2. manage module registry
      3. defines interface for reconfiguration of existing modules
      4. coordinate reconfiguration
   c. State management - what actions are allowed at what time?
      1. right now all we need to worry mostly about "RunSegments"
      2. define this as a state machine
   d. interact with Input Service
      (analysis involves deciding where the actual event loop lives)

   Inputs: CommandLineOption object (not needed immediately),
      Schedule Builder/Executor, InputModule, PSet
   Outputs: EventProcessor object

9. Context Factory and Context Framework and Context Building

   How does this Context come into existence? (1)
   How is it configured? (1)
   How is it managed?
   How do modules register callback function to be called when 
   part of the context changes?  

   The inputs and outputs of this task have not been completely
   identified yet.

   Inputs: EventProcessor
   Output: Mixin method of registering for callbacks

10. InputService

   A factory is needed for generating instances of input modules.
   Input service modules are distinct from worker-type modules.
   Some of the features below may be useful when doing data logging.

   a. Factory (1)
   b. Selecting specific products for inclusion/exclusion (3)
   c. Selecting based on trigger bits (3)
   d. making events invisible/visible (2)

   Inputs: InputService interface
   Outputs: Factory that makes instances of input modules

11. Output module additions

   To better support streams, the output modules will need to be enhanced
   to take advantage of routing information held within an event.

   a. Selecting specific products for inclusion/exclusion based on
      tagging (2)
   b. Selecting based on trigger bits (2)
   c. decide whether or not all or part of the tagging information will 
      persist (1)

   Inputs: tagging and trigger bit locating interface in the
      EventPrinicipal,
   Outputs: Utility classes for selecting products and events.

12. PSet 

   a. Local database management - storage of PSets with event data files
      <A>(2,12c)
   b. Local name lookup of PSets using in-memory cache
      <A>(1,12c)
   c. Generation of PSET_ID
      <A>(1,none)
   d. Empty 
   e. Global database design and implementation
      <A>(4,12a)
   f. Communication with global and other databases
      <A>(4,12e)
   g. updating of global database from local files
      <A>(4,12f)
   h. integration of help information with PSets
      <A>(3,none)
   i. enumeration of details of various tools for handling
      parameter sets and their configuration files
      <A>(4,none)
   j. create Path nodes from a stored PSet, i.e. generate a
      ProcessDesc object from a job PSet
      <A>(2,none)
   k. untracked parameter changing
      <A>(3,none)

   Outputs: PSet, ProcessDesc, parse tree Nodes, utilities

13. Tagging of events and EDProducts within an event

   Support for arbitrary tagging of products and events for
   routing purposes.  Output modules can select events and products
   based on this information.  Multiple tags are allowed per event
   or product.  This feature is useful in support of output streams.

   a. design interface for adding tags (temporary strings - lifetime of
      the event in memory) (1)
   b. design interface for selections based on tags (1)

   Inputs: Event,EventPrincipal,EDProduct

14. Keeping track of trigger bits (path results)

   The output modules will access events based on this information.

   a. paths must track decision information (1)
   b. products must be marked with trigger bits that caused its
      existence (2)
   c. output module must be capable of selection on this information (1)
   d. storage of this information (global scope, not temporary)
      within the event (1)
   e  extra configuration of filters, including pay attention to filter answer,
      ignore filter answer, or return opposite response of filter (1)

   Inputs: Worker
   Outputs: Trigger bit object 

15. Pre/Post worker and event loop functions

   Workers must allow for an arbitrary number of callback functions to
   be registered and invoked before and after the call to produce.
   Callback lists will also be needed for top and bottom of event loop.
   Support for functions shared amongst all modules is needed along 
   with unique functions per worker instance.  This is a general facility.

   a. arguments to callback functions need definition
      1. ModuleDescription?
      2. Collision ID?
   b. registration method needs to be established

   Inputs: Worker
   Output: utilities for calling all functions

16. Standard pre/post handlers

   Here is a list of tools that will be need and implemented using the
   pre/post function callback mechanism.

   a. Statistics gathering
      1. call counts to modules
      2. pass/fail counting per module/path
   b. Status of filters for HLT
   c. timing measurements (per event, per module)
   d. Simple memory leak checking
   e. Root-o-gram directory management

17. State change handler registration and management

   Module writers may be interested in getting calls when the
   "RunSegments" change.  We agreed that these handlers would be
   introduced using mixin classes.

   a. Define mixin class
   b. develop state interface / protocol between input service 
      and event process

18. Define Root-o-gram interaction service

   a. is this service needed?
   b. review DQM-flavor histogram interface and its applicability to
      all producers
   c. what other things like this service are necessary?

19. Thread Safety (this is in dispute as to whether or not it is a task)

   Of course all tasks must be careful to design their stuff to
   be thread safe - for example, don't use static variables to hold state.
   Questions of unique collision ID have come up in discusions about this
   task: is there more than one event with a particular collision ID?

   a. Protection of the Event during use of visualization tools
      or any other thing that needs multithreaded access to products
   b. protection when reconfiguring modules or interacting with them
   c. make changes and test them

   This item is delayed.

20. logging

   It seems we have identified three reasons why someone may want to
   send information to a log: (1) the program is reporting on progress
   and prints general information based on a verbosity level,
   (2) A situation has been determined that is not really an error
   (the code knows how to proceed - such as over time budget)
   and a message is logged indicating
   what type of situation it is and some information about it, and
   (3) An error occurred or was recognized (and possibly ignored) or
   some action was taken.

   a. identify the requirements for each of the above sitations
      and propose an API for each
   b. identify configuration needs.  How are actions associated with
      logging message specified?  How are they configured at runtime?
      How is the verbosity level set?
   c. propose a way to connect handlers to message type and allow program
      flow to be changed based on log messages received.
   d. "context" interface e.g. module name, event ID, runsegment, 
      etc. are automatically attached to logged messages.

21: external representation of logging data

   a. propose a way to handle externally representing the log 
      information.  There will need to be several external representations
      available for different areas: HLT, production, analysis, DQM.
      There are several external system that may be useful for this
      function including Apache's LogForCxx.
   b. Implement or integrate solutions

22. Exception processing

   The only way to alter the flow of control within the framework is
   by an exception throw.  Module code that cannot continue must throw
   an exception.  If log message handlers need to alter program flow,
   they will need to throw an exception and have it ripple back through
   the user's code (the code that logged the message).
   Each exception caught will be logged.
   The action or control flow change that occurs as a result of an
   exception catch is configurable and determined at runtime.
   The framework will know about Four classes of basic exceptions:
   (1) std::exception, (2) cms::exception, (3) edm::exception, and (4)
   anything else.

   a. determine the configuration options for directing control
      control
   b. determine the exception hierarchy and catch blocks within
      the framework.   Specify when and where each of the exception
      types are used.
   c. determine how configurable handlers will be dispatched on
      receipt of an exception by the program and the protocol
      between handlers and framework.
   d. determine what errors the EventProcessor handles and what it
      does not

23. Recording errors with event data

   When exceptions occur, products are likely to be missing from the
   event that are needed downstream.  The system should allow these
   events to be tagged, under user configuration control, so that
   output modules can decide what to do with them.

   a. decide where and how event tagging can take place.
   b. decide on configuration options (e.g. output module error streams,
      tag names, if tagging occurs).
   c. decide if error objects should be present in the event data
      to indicate what is missing and why.  Design and implement
      if this feature is needed



