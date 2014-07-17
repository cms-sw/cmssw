# Fasthadd RPM
I am documenting how to build a `RPM` for fasthadd in EL6.
I could not have succeeded in doing that without the [Fedora RPM guide](http://fedoraproject.org/wiki/How_to_create_an_RPM_package),
as well as the references therein.
Please read it: it is useful!

## A shortcut to build the rpm:
Do:
	$ rpmdev-setuptree

Checkout cmssw source anywhere (it has to include DQMServices/*) and run:
	$ ./makeRPM.sh ../../CMSSW_7_1_X_2014-07-15-0200/src/

Wait a bit, and done!.

## Getting the sources
If you want to build the `RPM` by yourself, create the `RPM` building directory structure

    $ rpmdev-setuptree

and copy them (`cp -p` preserves timestamps) into `~/rpmbuild/SOURCES`.

Next, you should download the `fasthadd archive` into the `~/rpmbuild/SOURCES` folder. This can be troublesome due to some GitHub features, so let me explain.

The Fedora suggestion is to use the following instructions in the `spec` file:

    %global commit a6cfbd2fad5011b6fdb08902efa3595cdc331e7e
    %global shortcommit %(c=%{commit}; echo ${c:0:7})
    Source0:        https://github.com/rovere/cmssw/archive/%{commit}/%{name}-%{version}-%{shortcommit}.tar.gz

but, investigating from [bug report](https://fedorahosted.org/fpc/ticket/284), the correct source is:

    https://github.com/rovere/cmssw/archive/%{commit}/%{commit}.tar.gz

Using `wget`, this will create a tarball file named `%{commit}`; if we untar it, we get a folder named `cmssw-%{commit}`, so the `%setup` command in the `%prep` section becomes

    %setup -q -n cmssw-%{commit}

In case of tags, we can use only

    %global tag fasthadd2.1

but this is not needed if we build tags using a combination of packages name and version, like `%{name}%{version}`.
In this case, we can use the "Releases" feature of GitHub; we can get the tarball of the tag using:

    https://github.com/rovere/cmssw/archive/%{name}%{version}.tar.gz

Using `wget`, this will create a tarball file named `%{name}%{version}`; if we untar it, we get a folder named `cmssw-%{name}%{version}`, so the `%setup` command in the `%prep` section becomes

    %setup -q -n cmssw-%{name}%{version}

In that bug report, another suggestion is to use as source

    https://github.com/rovere/cmssw/tarball/%{commit}

Using `wget`, this will create a tarball file named `%{commit}`; if we untar it, we get a folder named `rovere-cmssw-%{shortcommit}`, so the `%setup` command in the `%prep` section becomes

    %setup -q -n rovere-cmssw-%{shortcommit}

Finally, we can use GitHub API (currently `v3`): the archive fetching is documented [here](http://developer.github.com/v3/repos/contents/)
Assuming the `git` tag is `%{name}%{version}`, we can get the tarball from

    https://api.github.com/repos/rovere/cmssw/tarball/%{name}%{version}

Using `wget`, this will create a tarball file named `%{name}%{version}`; if we untar it, we get a folder named `rovere-cmssw-%{shortcommit}`, so the `%setup` command in the `%prep` section becomes

    %setup -q -n rovere-cmssw-%{shortcommit}

I use the release feature, so the command to get the tarball (since the timestamp checking are not supported because of a missing HTTP response header) are:

    wget -O %{name}%{version}.tar.gz https://github.com/rovere/cmssw/archive/%{name}%{version}.tar.gz

or

    curl https://github.com/rovere/cmssw/archive/%{name}%{version}.tar.gz > %{name}%{version}.tar.gz

## Getting the spec file
The spec file is in the `SPECS` subdirectory of this repo.
Give a look at it!
There are some conditions specific to EL5, due to the fact that some of the RPM macros are not supported in such arch.
Please note that I have put a condition in the 'Requires' section of the file:
the `ROOT` version you use for building should be the same as the one used for installation.
If you want to build the `RPM` by yourself, copy the spec file into `~/rpmbuild/SPECS`.

## Building the RPM
Note: Use a local dummy user for building RPMs! See the [Fedora RPM guide](http://fedoraproject.org/wiki/How_to_create_an_RPM_package) for more information.

Go into the `~rpmbuild/SPECS` directory, and validate the spec file:

    $ rpmlint fasthadd.spec

You should not get any errors (if so, I made something very wrong).
Now, you can build the `RPM`:

    $ rpmbuild -ba -vv fasthadd.spec

If the builds succeeds, you can find the `RPM` in `~/rpmbuild/RPMS`, and the source `RPM` in `~/rpmbuild/SRPMS`.

## Using the source RPM
The source `RPM` is very useful, since it ships all the source files, as well as the spec file, used for building a package.
The [Fedora RPM guide](http://fedoraproject.org/wiki/How_to_create_an_RPM_package) gives you more information, but you can simply run:

    $ rpm -ivh sourcepackage-name*.src.rpm

in order to install it into `~/rpmbuild`.
