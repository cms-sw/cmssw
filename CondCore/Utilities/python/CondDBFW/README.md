# CMSSW Release

The primary distribution channel is through CMSSW, https://github.com/cms-sw/cmssw.

The CMSSW releases will contain a version of CondDBFW that is stable for use inside CMSSW, and stable for use with web services.

# Web Service Releases

**You can download releases from gitlab, but these have not passed tests for inclusion into CMSSW.**

- **[See the release tags page of the gitlab repository for tagged releases.](https://gitlab.cern.ch/cms-ppdweb/cmsCondMiniFramework/tags)**
- **[Feature requests and issues](https://its.cern.ch/jira/browse/CMSCONDMIN/)**

# Notes on bugs in underlying libraries

**Note:** Due to what is apparently a bug in some part of the database framework CondDBFW uses, the following process done with CondDBFW may cause errors:

1. Connect to an oracle database (Prod/Prep, usually)
2. Get an object using connection.{object_name}(parameters)
3. Change a property of the object, eg tag.end_of_validity = 0
4. Get another object using the same connection

This may result in your change in step 3 being autocommitted to the database.  If you are working on Prod, in particular, this will probably cause a privilege error due to an attempt to write being made.
One way to solve this is to use connection.rollback(), a proxy method that just calls connection.session.rollback(), either before or after the failure to write to the oracle database.
If you are working in the python shell and the rollback does not work, the easiest way to get around it is to restart the shell.

## Object-Oriented Querying with CondDBFW

The first thing you usually do is to get some kind of entity that represents a row of a table in the conditions databases.  For example, you might require an object that represents a tag.  To do this, you should first setup a connection (and this is the only part of the framework that requires you to think about database connections - there isn't really a way around it, unless you use the framework's default connection, which is Frontier). Assuming you are working with the python shell, the code is as such:

```
>>> import CondCore.Utilities.CondDBFW.shell as shell
>>> connection = shell.connect(connection_string)
... if you used oracle, you are asked for your password if you didn't provide a secrets file ...
>>> tag = connection.tag(name="EBAlignment_hlt")
```

`tag` is now an instance of the `Tag` class, and has fields that have been filled with all data from the tag *EBAlignment_hlt*.  Below is a list of variations of this pattern, along with the data they result in:

1. `connection.tag(time_type="Run")` - `time_type` is not a primary key field in the tag table, and so this will usually not return a unique result.  In this case, you will get a list\* of `Tag` objects.

2. `connection.tag(name=["EBAlignment_hlt", "EBAlignment_measured_v01_express"])` - if you provide multiple values, this will usually determine non-unique results, and so this would return a list\* of `Tag` objects.
3. `connection.tag(insertion_time=range)` where `range = Range(datetimeA, datetimeB)`, and `Range` is at `CondDBFW.models.Range` - finds all Tags whose insertion_time falls within the interval $$[\text{datetimeA}, \text{datetimeB}]$$.  The parameters of `Range` are usually `datetime.datetime` or `int` objects.
4. `connection.tag(insertion_time=radius)` where `radius = Radius(datetimeA, timedelta)`, and `Radius` is at `CondDBFW.models.Radius` - finds all Tags whose insertion_time falls within the interval $$[\text{datetimeA-timedelta}, \text{datetimeB+timedelta}]$$.  The parameters of `Range` are usually `datetime.datetime` or `int` objects.
5. `connection.tag()` - creates an empty global tag with no data.  This is useful for executing methods defined on the object that do not require any data, such as `all()`.

\* not entirely true - you will get a `json_list` that contains the raw list of your tag objects.  This class is part of the `data_sources` module of the framework, and is there so you can do things like navigating json structures and printing tables much more easily.

## Setting up a Connection

For the Python shell, use ``CondDBFW.shell.connect`` and for Python scripts use ``CondDBFW.querying.connect``.  Each method takes the same parameters:

- ``connection_string`` - describe below.
- ``mode`` - either ``read`` or ``write`` - determines which entry in the netrc file is used.
- ``secrets`` - netrc secrets file.

If you are connecting to Frontier or SQLite, the connection strings are "frontier://database_name/schema" and "sqlite://file_name" respectively.

If you are connecting to Oracle, you can supply a connection string of the form "oracle://database_name/schema", in which case you have two options for supplying credentials:

- Let CondDBFW ask you for the username and password at runtime.  This is convenient for the Python shell environment, but not for scripting.
- Give a secrets netrc file as an argument to the connection constructor.  For example, `connection = shell.connect("oracle://cms_orcoff_prep/CMS_CONDITIONS", secrets="my_secrets_netrc_file")`.

# netrc File Format

The netrc file you give must have specific formats for its keys (the *machine* name for each entry):

```
machine database_name/schema/mode
    login username
    password password
```

The machine name is this way to allow you to have an account for each schema on each database, if you require it, inside one netrc file.


## Methods defined on Objects

Since the Conditions Databases are relational, we can easily define *parents* and *children* of objects that are constructed from Conditions Data.

For example, a *Tag* has *parent* *Global Tags*, and *child* *IOVs*.  There are methods in the framework to get all of this data - and they are listed below.

- `Global Tag`
  - ``tags(amount=10 [, search_criteria])`` - gets all tags belonging to this Global Tag (as GlobalTagMaps).
  - ``iovs(amount=10, valid=False [, search_criteria])`` - gets all IOVs that, if valid is True, all have insertion times strictly before the snapshot_time of this global tag.

- `Tag`
  - ``parent_global_tags( [, search_criteria])`` - gets all global tags to which this tag belongs.
  - ``all(amount=10 [, search_criteria])`` - can be executed on an empty tag (no column values are given when creating a tag).

- `IOV`
  - ``all(amount=10 [, search_criteria])`` - get all IOVs in the database.

- `Payload`
  - `parent_tags( [, search_criteria])` - get all tags that contain this payload.
  - `all(amount=10 [, search_criteria])` - gets all payloads in the database.

# Code for Common Use Cases

This page presents a list of common use cases, as well as code that can accomplish them.  The code is explained line by line.

All use cases for interaction with the framework in the python shell require a shell module instance, so the first thing you should do is run
```
import CondCore.Utilities.CondDBFW.shell as shell
con = shell.connect()
```

This imports the shell module from CondDBFW, and then connects Frontier.  Frontier doesn't require authentication, do neither the `mode` nor the account being used to connect needs to be given.

### Get IOVs inserted after a given time

To do this, we need the `Range` class, found at `CondDBFW.models.Range`:

```
>>> import datetime
>>> from CondCore.Utilities.CondDBFW.models import Range
```

We then define a start time for the range,

```
>>> start_time = datetime.datetime(year=2016, month=5, day=13, hour=14, minute=32)
```

And execute the query,

```
>>> connection.tag(name="JDAWES_ANOTHER_TEST")\
        .iovs(insertion_time=Range(start_time, datetime.datetime.now()))\
        .as_table()
```

This gives something similar to:

```
| TAG_NAME              | SINCE   | PAYLOAD_HASH                               | INSERTION_TIME        
| JDAWES_ANOTHER_TEST   | 1       | 1de84c791519dda91d8a967abf69974e21a08647   | 2016-05-13 14:32:10   
[...]
| JDAWES_ANOTHER_TEST   | 600     | ce1c90bf17e5ae445e8697705881fbf76d95772d   | 2016-05-13 14:32:10
```

### Get all Tags from Global Tag name and Record

To do this, we first get a list of `global_tag_map` objects by using the `global_tag_map()` proxy method defined on the `connection` object we got from `shell.connect()`, and giving the `global_tag_name` and `record`.  Since the combination of these values is not enough to form a selection on the composite primary key of the table, the result is a `json_list` of `global_tag_map` objects.

```
tag_maps = con.global_tag_map(global_tag_name="74X_dataRun1_HLT_frozen_v2", record="AlCaRecoTriggerBitsRcd")
```

We then need to get all Tag names from this data, so we use python's `map` function.  Notice that we have used `tag_maps.data()` to get the raw list of `tag_map` objects.

```
tag_names = map(lambda tag_map : tag_map.tag_name, tag_maps.data())
```

Since we now have a list of the names of tags that are associated with the (global_tag_name, record) pair we gave earlier, we can use this to select tags.  With CondDBFW, we can give the list as an argument to the proxy function for `tag` defined on the connection object.  Since a list is given, Tags are selected from the database based on each element of the list, and we obtain a `json_list` of `Tag` objects, which we then convert to a list of `dictionaries` with `as_dicts()`.  We could have also called `as_table()` on the result to draw a table in the python shell.  Refer to [JSON Data](json-data) to read about how to use the `as_table()` method.

```
tags = con.tag(name=tag_names).as_dicts()
```

### Get all Payloads belonging to a Tag

Assuming you have imported CondDBFW and you have a connection object called `con`, you can get all Payloads belonging to a specific Tag by doing the following:

First, get the tag object by supplying the tag name as a keyword argument to the proxy method defined on your connection.

```
 tag = con.tag(name="EBAlignment_hlt")
```

Then, get all the IOVs belonging to this tag with

```
iovs = tag.iovs()
```

Now we have the `json_list` of IOVs, we can map each IOV object to its payload hash, and then use this list of hashes to get payloads with each hash:

```
iov_hashes = map(lambda iov : iov.payload_hash, iovs.data())
payloads = con.payload(hash=iov_hashes)
```

# Type System

CondDBFW passes lists and dictionaries around using a type called `json_data_node` to make it easier for you to navigate and print data.

When you execute a query with CondDBFW, for example `con.tag().all()`, the value returned is a `json_list`.  This is a wrapper for a python `list`, but provides some things on top (including `as_table()`, `as_dicts()`, `data()` and `get_members()`, which are described below).

**Every querying pattern in CondDBFW returns a `json_list`, and quite a lot of features that involve processing data take a `json_data_node` as input - this means there is a standard data type across the whole framework, and all the code in CondDBFW can be sure that there are methods available that make a lot of operations simpler.**

**Tip: if you are unsure about what type your data is, you can use `json_data_node.make(your_data)` to make the appropriate wrapper object for your data.**

### Using `data()` to get raw data from queries

Since CondDBFW queries return either None, `json_list` or an ORM object, if your query returns `json_list`, you can use `data()` to get the python primitive data type.  For example, `tags = con.tag().all().data()` gives a python `list` object, whereas omitting `data()` gives a `json_list` object.

**Note: `json_list` objects are iterable, so can be passed as arguments if the argument in question must be iterable.**

### Using `get_members()` to get a single member of each object

`get_members()` is an object-oriented implementation of the python `map` function.  It simply takes a `json_list` of CondDBFW objects, and returns the `json_list` consisting of the member specified of each CondDBFW object in the original `json_list`.  For example, `con.tag(name="EBAlignment_hlt").iovs().get_members("payload_hash").data()` simply returns a list of payload hashes used by IOVs in the tag *EBAlignment_hlt*.

### Returning JSON with `as_dicts()`

The `as_dicts()` method is defined on `json_list` objects and on individual CondDBFW objects returned from queries.  Executing this method on an object will yield the object's dictionary, and executing it on a `json_list` will yield a list of dictionaries, in which case one can use `json.dumps()` to get the valid JSON string representation of the list.

### Printing tables with `as_table()`

The `as_table()` method provided by the framework computes widths of columns, and allows you to choose which columns to draw and how to draw them.  This part of the framework will be developed further.

`as_table()` is defined on `json_list`, and is designed to be used when you use a method that returns a list of objects.  For example, a common pattern might be to use `connection.tag().all().as_table()`, since `all()` returns a json_list.  Note that the json_list is converted to a list of dictionaries (no longer inside a json_list wrapper).

A list of parameters that as_table() takes (none of which are compulsory) is given below.

- `fit` - a `list` of columns whose width should be precisely calculated to fit the width of the longest content.
- `columns` - a `list` of columns to be shown.  Any columns not included will not be shown.
- `hide` - a `list` of columns to be hidden.
- `col_width` - `int` - the column width that computations are biassed towards when column widths are being computed based on fit, columns and hide.
- `row_nums` - `boolean` - adds a column for row numbers.

Note: If you want to print a table of data that has not been returned from one of the framework's methods, you should use the format `[{"col" : "value", ..., "col" : "value"}, ..., {"col" : "value", ..., "col" : "value"}]`.