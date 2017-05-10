# Build -python subpackage
%bcond_without python

# Build -emacs and -emacs-el (except on EL5 and EL6)
%if 0%{?el5} || 0%{?el6}
%bcond_with emacs
%else
%bcond_without emacs
%endif

# Build -java subpackage (except on EL5 and EL6)
%if 0%{?el5} || 0%{?el6}
%bcond_with java
%else
%bcond_without java
%endif

# Don't require gtest
%bcond_with gtest

%if %{with python}
%define python_sitelib %(%{__python} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()")
%endif

%if %{with emacs}
%global emacs_version %(pkg-config emacs --modversion)
%global emacs_lispdir %(pkg-config emacs --variable sitepkglispdir)
%global emacs_startdir %(pkg-config emacs --variable sitestartdir)
%endif

Summary:        Protocol Buffers - Google's data interchange format
Name:           protobuf
Version:        2.4.1
Release:        13%{?dist}
License:        BSD
Group:          Development/Libraries
Source:         http://protobuf.googlecode.com/files/%{name}-%{version}.tar.bz2
Source1:        ftdetect-proto.vim
%if %{with emacs}
Source2:        protobuf-init.el
%endif
Patch1:         protobuf-2.3.0-fedora-gtest.patch
Patch2:         protobuf-2.4.1-java-fixes.patch
URL:            http://code.google.com/p/protobuf/
%if 0%{?el5}
BuildRoot:      %(mktemp -ud %{_tmppath}/%{name}-%{version}-%{release}-XXXXXX)
%endif
BuildRequires:  automake autoconf libtool pkgconfig zlib-devel

%if %{with emacs}
BuildRequires:  emacs
BuildRequires:  emacs-el >= 24.1
%endif

%if %{with gtest}
BuildRequires:  gtest-devel
%endif

%description
Protocol Buffers are a way of encoding structured data in an efficient
yet extensible format. Google uses Protocol Buffers for almost all of
its internal RPC protocols and file formats.

Protocol buffers are a flexible, efficient, automated mechanism for
serializing structured data – think XML, but smaller, faster, and
simpler. You define how you want your data to be structured once, then
you can use special generated source code to easily write and read
your structured data to and from a variety of data streams and using a
variety of languages. You can even update your data structure without
breaking deployed programs that are compiled against the "old" format.

%package compiler
Summary: Protocol Buffers compiler
Group: Development/Libraries
Requires: %{name} = %{version}-%{release}

%description compiler
This package contains Protocol Buffers compiler for all programming
languages

%package devel
Summary: Protocol Buffers C++ headers and libraries
Group: Development/Libraries
Requires: %{name} = %{version}-%{release}
Requires: %{name}-compiler = %{version}-%{release}
Requires: pkgconfig

%description devel
This package contains Protocol Buffers compiler for all languages and
C++ headers and libraries

%package static
Summary: Static development files for %{name}
Group: Development/Libraries
Requires: %{name} = %{version}-%{release}

%description static
Static libraries for Protocol Buffers

%package lite
Summary: Protocol Buffers LITE_RUNTIME libraries
Group: Development/Libraries

%description lite
Protocol Buffers built with optimize_for = LITE_RUNTIME.

The "optimize_for = LITE_RUNTIME" option causes the compiler to generate code
which only depends libprotobuf-lite, which is much smaller than libprotobuf but
lacks descriptors, reflection, and some other features.

%package lite-devel
Summary: Protocol Buffers LITE_RUNTIME development libraries
Group: Development/Libraries
Requires: %{name}-devel = %{version}-%{release}
Requires: %{name}-lite = %{version}-%{release}

%description lite-devel
This package contains development libraries built with
optimize_for = LITE_RUNTIME.

The "optimize_for = LITE_RUNTIME" option causes the compiler to generate code
which only depends libprotobuf-lite, which is much smaller than libprotobuf but
lacks descriptors, reflection, and some other features.

%package lite-static
Summary: Static development files for %{name}-lite
Group: Development/Libraries
Requires: %{name}-devel = %{version}-%{release}

%description lite-static
This package contains static development libraries built with
optimize_for = LITE_RUNTIME.

The "optimize_for = LITE_RUNTIME" option causes the compiler to generate code
which only depends libprotobuf-lite, which is much smaller than libprotobuf but
lacks descriptors, reflection, and some other features.

%if %{with python}
%package python
Summary: Python bindings for Google Protocol Buffers
Group: Development/Languages
BuildRequires: python-devel
Conflicts: %{name}-compiler > %{version}
Conflicts: %{name}-compiler < %{version}

%if 0%{?el5}
BuildRequires: python-setuptools
%else
BuildRequires: python-setuptools-devel
%endif

%description python
This package contains Python libraries for Google Protocol Buffers
%endif

%package vim
Summary: Vim syntax highlighting for Google Protocol Buffers descriptions
Group: Development/Libraries
Requires: vim-enhanced

%description vim
This package contains syntax highlighting for Google Protocol Buffers
descriptions in Vim editor

%if %{with emacs}
%package emacs
Summary: Emacs mode for Google Protocol Buffers descriptions
Group: Applications/Editors
Requires: emacs >= 0%{emacs_version}

%description emacs
This package contains syntax highlighting for Google Protocol Buffers
descriptions in the Emacs editor.

%package emacs-el
Summary: Elisp source files for Google protobuf Emacs mode
Group: Applications/Editors
Requires: protobuf-emacs = %{version}

%description emacs-el
This package contains the elisp source files for %{pkgname}-emacs
under GNU Emacs. You do not need to install this package to use
%{pkgname}-emacs.
%endif

%if %{with java}
%package java
Summary: Java Protocol Buffers runtime library
Group:   Development/Languages
BuildRequires:    java-devel >= 1.6
BuildRequires:    jpackage-utils
BuildRequires:    maven-local
BuildRequires:    maven-compiler-plugin
BuildRequires:    maven-install-plugin
BuildRequires:    maven-jar-plugin
BuildRequires:    maven-javadoc-plugin
BuildRequires:    maven-resources-plugin
BuildRequires:    maven-surefire-plugin
BuildRequires:    maven-antrun-plugin
Requires:         java
Requires:         jpackage-utils
Conflicts:        %{name}-compiler > %{version}
Conflicts:        %{name}-compiler < %{version}

%description java
This package contains Java Protocol Buffers runtime library.

%package javadoc
Summary: Javadocs for %{name}-java
Group:   Documentation
Requires: jpackage-utils
Requires: %{name}-java = %{version}-%{release}

%description javadoc
This package contains the API documentation for %{name}-java.

%endif

%prep
%setup -q
%if %{with gtest}
rm -rf gtest
%patch1 -p1 -b .gtest
%endif
chmod 644 examples/*
%if %{with java}
%patch2 -p1 -b .java-fixes
rm -rf java/src/test
%endif

%build
iconv -f iso8859-1 -t utf-8 CONTRIBUTORS.txt > CONTRIBUTORS.txt.utf8
mv CONTRIBUTORS.txt.utf8 CONTRIBUTORS.txt
export PTHREAD_LIBS="-lpthread"

# don't regen on el5 (cause we are patching configure)
%if ! 0%{?el5}
./autogen.sh
%endif

%configure

make %{?_smp_mflags}

%if %{with python}
pushd python
python ./setup.py build
sed -i -e 1d build/lib/google/protobuf/descriptor_pb2.py
popd
%endif

%if %{with java}
pushd java
mvn-rpmbuild install javadoc:javadoc
popd
%endif

%if %{with emacs}
emacs -batch -f batch-byte-compile editors/protobuf-mode.el
%endif

%check
make %{?_smp_mflags} check

%install
rm -rf %{buildroot}
make %{?_smp_mflags} install DESTDIR=%{buildroot} STRIPBINARIES=no INSTALL="%{__install} -p" CPPROG="cp -p"
find %{buildroot} -type f -name "*.la" -exec rm -f {} \;

%if %{with python}
pushd python
python ./setup.py install --root=%{buildroot} --single-version-externally-managed --record=INSTALLED_FILES --optimize=1
popd
%endif
install -p -m 644 -D %{SOURCE1} %{buildroot}%{_datadir}/vim/vimfiles/ftdetect/proto.vim
install -p -m 644 -D editors/proto.vim %{buildroot}%{_datadir}/vim/vimfiles/syntax/proto.vim

%if %{with java}
pushd java
install -d -m 755 %{buildroot}%{_javadir}
install -pm 644 target/%{name}-java-%{version}.jar %{buildroot}%{_javadir}/%{name}.jar

install -d -m 755 %{buildroot}%{_javadocdir}/%{name}
cp -rp target/site/apidocs %{buildroot}%{_javadocdir}/%{name}

install -d -m 755 %{buildroot}%{_mavenpomdir}
install -pm 644 pom.xml %{buildroot}%{_mavenpomdir}/JPP-%{name}.pom
%add_maven_depmap JPP-%{name}.pom %{name}.jar
popd
%endif

%if %{with emacs}
mkdir -p $RPM_BUILD_ROOT%{emacs_lispdir}
mkdir -p $RPM_BUILD_ROOT%{emacs_startdir}
install -p -m 0644 editors/protobuf-mode.el $RPM_BUILD_ROOT%{emacs_lispdir}
install -p -m 0644 editors/protobuf-mode.elc $RPM_BUILD_ROOT%{emacs_lispdir}
install -p -m 0644 %{SOURCE2} $RPM_BUILD_ROOT%{emacs_startdir}
%endif

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%post lite -p /sbin/ldconfig
%postun lite -p /sbin/ldconfig

%post compiler -p /sbin/ldconfig
%postun compiler -p /sbin/ldconfig

%if 0%{?el5}
%clean
rm -rf %{buildroot}
%endif

%files
%defattr(-, root, root, -)
%{_libdir}/libprotobuf.so.*
%doc CHANGES.txt CONTRIBUTORS.txt COPYING.txt README.txt

%files compiler
%defattr(-, root, root, -)
%{_bindir}/protoc
%{_libdir}/libprotoc.so.*
%doc COPYING.txt README.txt

%files devel
%defattr(-, root, root, -)
%dir %{_includedir}/google
%{_includedir}/google/protobuf/
%{_libdir}/libprotobuf.so
%{_libdir}/libprotoc.so
%{_libdir}/pkgconfig/protobuf.pc
%doc examples/add_person.cc examples/addressbook.proto examples/list_people.cc examples/Makefile examples/README.txt

%files static
%defattr(-, root, root, -)
%{_libdir}/libprotobuf.a
%{_libdir}/libprotoc.a

%files lite
%defattr(-, root, root, -)
%{_libdir}/libprotobuf-lite.so.*

%files lite-devel
%defattr(-, root, root, -)
%{_libdir}/libprotobuf-lite.so
%{_libdir}/pkgconfig/protobuf-lite.pc

%files lite-static
%defattr(-, root, root, -)
%{_libdir}/libprotobuf-lite.a

%if %{with python}
%files python
%defattr(-, root, root, -)
%dir %{python_sitelib}/google
%{python_sitelib}/google/protobuf/
%{python_sitelib}/protobuf-%{version}-py2.?.egg-info/
%{python_sitelib}/protobuf-%{version}-py2.?-nspkg.pth
%doc python/README.txt
%doc examples/add_person.py examples/list_people.py examples/addressbook.proto
%endif

%files vim
%defattr(-, root, root, -)
%{_datadir}/vim/vimfiles/ftdetect/proto.vim
%{_datadir}/vim/vimfiles/syntax/proto.vim

%if %{with emacs}
%files emacs
%defattr(-,root,root,-)
%{emacs_startdir}/protobuf-init.el
%{emacs_lispdir}/protobuf-mode.elc

%files emacs-el
%defattr(-,root,root,-)
%{emacs_lispdir}/protobuf-mode.el
%endif

%if %{with java}
%files java
%defattr(-, root, root, -)
%{_mavenpomdir}/JPP-%{name}.pom
%{_mavendepmapfragdir}/%{name}
%{_javadir}/%{name}.jar
%doc examples/AddPerson.java examples/ListPeople.java

%files javadoc
%defattr(-, root, root, -)
%{_javadocdir}/%{name}
%endif

%changelog
* Fri Feb 07 2014 Salvatore Di Guida <salvatore.di.guida[at]cern.ch> - 2.4.1-13
- Ported to el6.
- do not build emacs subpackage for el6.

* Tue Feb 26 2013 Conrad Meyer <cemeyer@uw.edu> - 2.4.1-12
- Nuke BR on maven-doxia, maven-doxia-sitetools (#915620)

* Thu Feb 14 2013 Fedora Release Engineering <rel-eng@lists.fedoraproject.org> - 2.4.1-11
- Rebuilt for https://fedoraproject.org/wiki/Fedora_19_Mass_Rebuild

* Wed Feb 06 2013 Java SIG <java-devel@lists.fedoraproject.org> - 2.4.1-10
- Update for https://fedoraproject.org/wiki/Fedora_19_Maven_Rebuild
- Replace maven BuildRequires with maven-local

* Sun Jan 20 2013 Conrad Meyer <konrad@tylerc.org> - 2.4.1-9
- Fix packaging bug, -emacs-el subpackage should depend on -emacs subpackage of
  the same version (%%version), not the emacs version number...

* Thu Jan 17 2013 Tim Niemueller <tim@niemueller.de> - 2.4.1-8
- Added sub-package for Emacs editing mode

* Sat Jul 21 2012 Fedora Release Engineering <rel-eng@lists.fedoraproject.org> - 2.4.1-7
- Rebuilt for https://fedoraproject.org/wiki/Fedora_18_Mass_Rebuild

* Mon Mar 19 2012 Dan Horák <dan[at]danny.cz> - 2.4.1-6
- disable test-suite until g++ 4.7 issues are resolved

* Mon Mar 19 2012 Stanislav Ochotnicky <sochotnicky@redhat.com> - 2.4.1-5
- Update to latest java packaging guidelines

* Tue Feb 28 2012 Fedora Release Engineering <rel-eng@lists.fedoraproject.org> - 2.4.1-4
- Rebuilt for c++ ABI breakage

* Sat Jan 14 2012 Fedora Release Engineering <rel-eng@lists.fedoraproject.org> - 2.4.1-3
- Rebuilt for https://fedoraproject.org/wiki/Fedora_17_Mass_Rebuild

* Tue Sep 27 2011 Pierre-Yves Chibon <pingou@pingoured.fr> - 2.4.1-2
- Adding zlib-devel as BR (rhbz: #732087)

* Thu Jun 09 2011 BJ Dierkes <wdierkes@rackspace.com> - 2.4.1-1
- Latest sources from upstream.
- Rewrote Patch2 as protobuf-2.4.1-java-fixes.patch

* Wed Feb 09 2011 Fedora Release Engineering <rel-eng@lists.fedoraproject.org> - 2.3.0-7
- Rebuilt for https://fedoraproject.org/wiki/Fedora_15_Mass_Rebuild

* Thu Jan 13 2011 Stanislav Ochotnicky <sochotnicky@redhat.com> - 2.3.0-6
- Fix java subpackage bugs #669345 and #669346
- Use new maven plugin names
- Use mavenpomdir macro for pom installation

* Mon Jul 26 2010 David Malcolm <dmalcolm@redhat.com> - 2.3.0-5
- generalize hardcoded reference to 2.6 in python subpackage %%files manifest

* Wed Jul 21 2010 David Malcolm <dmalcolm@redhat.com> - 2.3.0-4
- Rebuilt for https://fedoraproject.org/wiki/Features/Python_2.7/MassRebuild

* Thu Jul 15 2010 James Laska <jlaska@redhat.com> - 2.3.0-3
- Correct use of %%bcond macros

* Wed Jul 14 2010 James Laska <jlaska@redhat.com> - 2.3.0-2
- Enable python and java sub-packages

* Tue May 4 2010 Conrad Meyer <konrad@tylerc.org> - 2.3.0-1
- bump to 2.3.0

* Wed Sep 30 2009 Lev Shamardin <shamardin@gmail.com> - 2.2.0-2
- added export PTHREAD_LIBS="-lpthread"

* Fri Sep 18 2009 Lev Shamardin <shamardin@gmail.com> - 2.2.0-1
- Upgraded to upstream protobuf-2.2.0
- New -lite packages

* Sun Mar 01 2009 Caolán McNamra <caolanm@redhat.com> - 2.0.2-8
- add stdio.h for sprintf, perror, etc.

* Thu Feb 26 2009 Fedora Release Engineering <rel-eng@lists.fedoraproject.org> - 2.0.2-7
- Rebuilt for https://fedoraproject.org/wiki/Fedora_11_Mass_Rebuild

* Tue Dec 23 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.2-6
- Small fixes for python 2.6 eggs.
- Temporarily disabled java subpackage due to build problems, will be fixed and
  turned back on in future.

* Thu Nov 27 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.2-5
- No problems with ppc & ppc64 arch in rawhide, had to do a release bump.

* Sat Nov 22 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.2-4
- Added patch from subversion r70 to workaround gcc 4.3.0 bug (see
  http://code.google.com/p/protobuf/issues/detail?id=45 for more
  details).

* Tue Nov 11 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.2-3
- Added conflicts to java and python subpackages to prevent using with
  wrong compiler versions.
- Fixed license.
- Fixed BuildRequires for -python subpackage.
- Fixed Requires and Group for -javadoc subpackage.
- Fixed Requires for -devel subpackage.
- Fixed issue with wrong shebang in descriptor_pb2.py.
- Specify build options via --with/--without.
- Use Fedora-packaged gtest library instead of a bundled one by
  default (optional).

* Fri Oct 31 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.2-2
- Use python_sitelib macro instead of INSTALLED_FILES.
- Fix the license.
- Fix redundant requirement for -devel subpackage.
- Fix wrong dependences for -python subpackage.
- Fix typo in requirements for -javadoc subpackage.
- Use -p option for cp and install to preserve timestamps.
- Remove unneeded ldconfig call for post scripts of -devel subpackage.
- Fix directories ownership.

* Sun Oct 12 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.2-1
- Update to version 2.0.2
- New -java and -javadoc subpackages.
- Options to disable building of -python and -java* subpackages

* Mon Sep 15 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.1-2
- Added -p switch to install commands to preserve timestamps.
- Fixed Version and Libs in pkgconfig script.
- Added pkgconfig requires for -devel package.
- Removed libtool archives from -devel package.

* Thu Sep 04 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.1-1
- Updated to 2.0.1 version.

* Wed Aug 13 2008 Lev Shamardin <shamardin@gmail.com> - 2.0.0-0.1.beta
- Initial package version. Credits for vim subpackage and pkgconfig go
  to Rick L Vinyard Jr <rvinyard@cs.nmsu.edu>
