#  FWCore/Concurrency Description

## Introduction
The classes in FWCore/Concurrency build upon Intel's Threading Building Blocks library to provide a richer set of concurrency classes to express a wider set of asynchronous patterns. Below are descriptions of the various parts of the library.

## `edm::TaskBase`
This class represents a set of code to be run via the member function overload `void execute()`. The class includes an intrusive reference count (`increment_ref_count()` and `decrement_ref_count()`) to allow for _join_ semantics (i.e. the ability to wait until a group of concurrent work has finished before executing the task) by having each task to be waited upon increment the ref count and once finished, decrement it. 

Exactly how to _dispose_ of the class can be customized via the virtual function `void recycle()` which defaults to `delete this;`. The class `edm::TaskSentry` is meant to be used as a guard to properly handle the lifetime of an `edm::TaskBase` by calling `recycle()` during the `edm::TaskSentry`'s destructor.

## `edm::WaitingTask`
This class inherits from `edm::TaskBase` and represents a set of code to be run once other activities have completed. This includes the ability to hold a `std::exception_ptr` which can hold an exception which was generated in a dependent task.

A raw pointer to a  `edm::WaitingTask` is not supposed to be handled directly. Instead, one should use the helpers `edm::WaitingTaskHolder`, `edm::WaitingTaskWithArenaHolder` or `edm::WaitingTaskList` to properly manage the internal reference count such that when the count drops to 0 the `execute()` method will be run followed by `recycle()`. In addition, these helper classes will handled passing along any `edm::exception_ptr` generated from a dependent task.

The easiest way to create an `edm::WaitingTask` is to call `edm::make_waiting_task` and pass in a lambda of the form `void(std::exception_ptr const*)`.
```C++
  tbb::task_group group;
  auto task = edm::make_waiting_task([](std::exception_ptr const* iPtr) { if(not iPtr) { runCalculation(); }});
  edm::WaitingTaskHolder taskHolder{ group, task };
```

### `edm::FinalWaitingTask`
In the case where one is doing a synchronous wait on a series of asynchronous tasks, it is useful to have a special `edm::WaitingTask` that can sit on the stack and hold onto any `std::exception_ptr` which occur during the asynchronous processing.

```C++
   edm::FinalWaitingTask finalTask;
   
   tbb::task_group group;
   doLotsOfWorkAsynchronously( edm::WaitingTaskHolder(group, &finalTask) );
   
   group.wait();
   assert(finalTask.done());
   if(auto excptPtr = finalTask.exceptionPtr()) {
    std::throw_exception( *excptPtr );
   }
```
## `edm::WaitingTaskHolder`
This class functions as a _smart pointer_ for an `edm::WaitingTask`. On construction it will increment the embedded reference count of the `edm::WaitingTask`. On either the destructor or the call to `doneWaiting(std::exception_ptr)` it will decrement the reference count. If the count goes to 0, the `edm::WaitingTaskHolder` will pass the `std::exception_ptr` onto the `edm::WaitingTask`, call `execute()` under the `tbb:task_group` given to the holder and finally call `recycle()`.

It is best to pass an `edm::WaitingTaskHolder` by value as the copy and move operators properly handle the reference counting of the held `edm::WaitingTask`.

A standard idiom is to pass a `edm::WaitingTaskHolder` to an `edm::WaitingTask` or to another function to explicitly create a chain of tasks to execute.
```C++

  void workAsync(edm::WaitingTaskHolder lastTask) {
  edm::WaitingTaskHolder nextTask( 
    lastTask.group(), 
    edm::make_waiting_task([&ptr, task = stdm::move(lastTask) ](std::exception_ptr const* iExcept) mutable {
      if(not iExcept) {
        doTheNextWork();
      } else {
        //pass the exception to the next task
        task.doneWaiting(*iExcept);
      }
      //destructor of task will automatically call execute on the underlying edm::WaitingTask
    }));

  doEvenMoreWorkAsync( std::move(nextTask) );
}
```

## WaitingTask Chains
As seen, one often wants to run a task as the last step of another task. Composing such a _chain_ of tasks is made easier via the functions in the `edm::waiting_task::chain` namespace. Similar to the C++20 _range_ library, the functions in this namespace are composed via the use of `operator|`.

NOTE: in all following code snippets `using namespace edm::waiting_task` should be inferred.

### `first` and `last` of a chain
A chain is begun by calling `chain::first` and passing in a lambda that takes a `edm::WaitingTaskHolder`.


A chain can end in one of two ways. One is a call to `chain::lastTask` which takes a `edm::WaitingTaskHolder` and results in the `operator|` chain returning a new `edm::WaitingTaskHolder` corresponding to the first task in the chain.
```C++
  edm::WaitingTaskHolder composeDoFirstAsync(edm::WaitingTaskHolder nextTask) {
    return chain::first([](auto nextTask) { doFirst(std::move(nextTask)); } )
           | chain::lastTask(std::move(nextTask));
  }
```
The other way is to call `chain::runLast` which will cause the first task to be run via `tbb::task_group::run()`.
```C++
  void doFirstAsync(edm::WaitingTaskHolder nextTask) {
    chain::first([](auto nextTask) { doFirst(std::move(nextTask)); } )
    | chain::runLast(std::move(nextTask));
  }
```

If you want full control over how exceptions are to be handled, you can pass in a functor of the form `void(std::exception_ptr const*, edm::WaitingTaskHolder)`.

In the following example, the `edm::WaitingTaskHolder` returned from the function might be passed an exception by calling its `doneWaiting()`. If so, the exception will be printed but not passed on to the next tasks in the chain.
```C++
  edm::WaitingTaskHolder composeDoFirstAsync(edm::WaitingTaskHolder nextTask) {
    return chain::first([](std::exception_ptr const* iPtr, auto nextTask) { 
                         if(iPtr) { printExceptionAndIgnore(*iPtr); }
                         doFirst(std::move(nextTask)); } )
           | chain::lastTask(std::move(nextTask));
  }
```

### chaining tasks with `then`

One can chain multiple tasks together using the `chain::then` function
```C++
 void doStuffAsync(edm::WaitingTaskHolder lastTask) {
    chain::first([](auto nextTask){ do_a_async(nextTask); })
    | chain::then([](auto nextTask){ do_b_async(nextTask); })
    | chain::then([](auto nextTask){ do_c_async(nextTask); })
    | chain::runLast(std::move(lastTask));
 }
```
If an exception occurs in an earlier task, the functor in the following `chain::then` will not be run and the exception will be propagated through the chain to the `edm::WaitingTaskHolder` passed to `runLast` or `lastTask`.

Similarly to `chain::first` you can explicitly handle any exception. In the following, if `do_a_async` has an exception, the exception will be printed and then discarded so that `do_b_async` will be called and `lastTask` will not see the exception.
```C++
void doStuffAsync(edm::WaitingTaskHolder lastTask) {
   chain::first([](auto nextTask){ do_a_async(nextTask); })
   | chain::then([](std::exception_ptr const* iPtr, auto nextTask){ 
     if(iPtr) {
      printExceptionAndIgnore(*iPtr);
     }
     do_b_async(nextTask); 
    })
   | chain::runLast(std::move(lastTask));
}
```

### conditional task with `ifThen`
One can decide to add or not add a task to a chain via the `chain::ifThen` function. The conditional is evaluated at the time of the call to `ifThen`.

In the following, `doVerboseDump()` will only be called if `isVerbose==true` at the time the chain is being composed.
```C++
  chain::first([](auto nextTask){ doWorkAsync(nextTask); })
  | chain::ifThen(isVerbose, [](auto nextTask){ doVerboseDump();})
  | chain::runLast(lastTask);
```

### helper `ifException().else_()`
Use the helper in the case where one needs to do something special when an earlier task had an exception but still want the exception propagated to the next task in the chain and do not want the work to be done for this task if an exception is thrown. The helper can be used in place of the lambda passed to any `chain` function.

In the following, if an exception is passed to the result of this function, `printException` will be called but `doFirstIfNoException` will not be called. In addition, the exception will be propagated to `nextTask`.
```C++
  edm::WaitingTaskHolder composeDoFirstAsync(edm::WaitingTaskHolder nextTask) {
    return chain::first(chain::ifException([](auto except) { printException(except); })
                               .else_([](auto nextTask) { doFirstIfNoException(std::move(nextTask)); } )
           | chain::lastTask(std::move(nextTask));
  }
```

## `edm::WaitingTaskList`
This class is similar to `edm::WaitingTaskHolder` except this class can hold onto many `edm::WaitingTask`s. New tasks can be added to the list via a call to `add()`. Concurrent `add()` calls are thread safe.

Calling `doneWaiting(std::exception_ptr)` will cause the class to decrement the reference count of all held `edm::WaitingTask`s and if their count is 0 it will then call `execute()` and `recycle()` via the passed in `edm::task_group`. If a non-default `std::exception_ptr` is passed to `doneWaiting()` that `std::exception_ptr` will be passed to all held `edm::WaitingTask`s. If further `edm::WaitingTask`s are `add()`ed to the class after `doneWaiting()` was called, those new `edm::WaitingTask`s will have their reference counts immediately decremented and then the standard procedure for reaching 0 will be enacted.

An `edm::WaitingTaskList` can be used multiple times by calling the `reset()` method. This method must only be called after `doneWaiting()` has been called AND when no further `add()` will be called which are supposed to be associated with the previous `doneWaiting()` call.

## `edm::WaitingTaskWithArenaHolder`
This class behaves just like `edm::WaitingTaskHolder` except it will use the `tbb::task_arena` is is given when calling `tbb::task_group::run` rather than using the default `tbb::task_arena` associated with the local thread. This is useful for the case where one wants to potentially enqueue a task from a non-TBB thread.

## `edm::SerialTaskQueue`
One needs to serialize access to non-thread-safe shared resources. Rather than using a thread blocking primative, like a mutex, one can use the `edm::SerialTaskQueue`. The class guarantees that one and only one task from the queue will be executing at any given time. The tasks are run asynchronously.

A task is added to the queue via the call to `push(tbb::task_group&, F&&)` where the second argument is a lambda of the form `void()`. If no other task from the queue is running during the call to `push()` then the task will immediately be passed to `tbb::task_group::run`. If a task is already running, the new task will be placed at the end of the list of presently waiting tasks. Once the running task completes, it will automatically all `tbb::task_group::run` on the longest waiting task. Concurrent calls to `push()` are safe.

The action of pulling a waiting task off the queue and running it can be paused by calling `pause()`. The queue can be restarted by calling `resume()`. Multiple `pause()` calls can be made just as long as an equal number of `resume()` calls.

Example: protecting `std::cout` so printouts do not intertwine.

```C++
  edm::SerialTaskQueue queue;
  
  tbb::task_group group;
  for(int i=0; i<3; ++i) {
    group.run([&queue, &group] {
      usleep(1000);
      queue.push(group1, [i](){
        std::cout <<"loop 1"<<i<<"\n";
      });
    });
  }
  
  for(int i=0; i<6; ++i) {
    group.run([&queue, &group] {
      usleep(1500);
      queue.push(group1, [i](){
        std::cout <<"loop 2"<<i<<"\n";
      });
    });
  }
  
  group.wait();
```

## `edm::SerialTaskQueueChain`
If multiple non-thread-safe shared resources need to be acquired then one should use a `edm::SerialTaskQueueChain`. The call to `push()` will guarantee that all the `edm::SerialTaskQueue`s are `push`ed and `pause`d in the order in which they were passed to the constructor before the task is run. This guarantees that the running task is the only task that _owns_ all the resources associated with the queues.

NOTE: if the individual `edm::SerialTaskQueue`s are also used by other `edm::SerialTaskQueueChain`s then the relative order of the `edm::SerialTaskQueue`s passed to all the chains MUST be the same between all the chains. This is required to avoid the possiblity of a deadlock. (See information about the Monitor pattern for concurrent programming).

## `edm::LimitedTaskQueue`
This class allows N tasks from the queue to be run at any given time where N in an integer value passed to the constructor of the class. This class is useful in the case where one want to set a bounds on the use of some concurrent capable resources.

```C++
  //limit the amount of memory used to 8GB by having at most 8 task running each with 1GB
  edm::LimitedTaskQueue queue(8);
  
  tbb::task_group group;
  for(int i=0; i< 100; ++i) {
    group.run([&group, &queue] () {
      auto result = doWork();
      
      queue.push(group, [result=std::move(result)]() {
        //need lots of memory for next part
        std::unique_ptr<int[]> buffer( new int[1000*1000*1000/sizeof(int)]);
        lastPartOfWork(std::move(result), std::move(buffer));
      });
      
    });
  }
  
  group.wait();
}
```
