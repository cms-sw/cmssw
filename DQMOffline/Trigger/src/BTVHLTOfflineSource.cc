


<!DOCTYPE html>
<html lang="en" class="">
  <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# object: http://ogp.me/ns/object# article: http://ogp.me/ns/article# profile: http://ogp.me/ns/profile#">
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta http-equiv="Content-Language" content="en">
    <meta name="viewport" content="width=1020">
    
    
    <title>cmssw/BTVHLTOfflineSource.cc at BTVHLTDQMOffline_fix · silviodonato/cmssw · GitHub</title>
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
    <link rel="apple-touch-icon" sizes="57x57" href="/apple-touch-icon-114.png">
    <link rel="apple-touch-icon" sizes="114x114" href="/apple-touch-icon-114.png">
    <link rel="apple-touch-icon" sizes="72x72" href="/apple-touch-icon-144.png">
    <link rel="apple-touch-icon" sizes="144x144" href="/apple-touch-icon-144.png">
    <meta property="fb:app_id" content="1401488693436528">

      <meta content="@github" name="twitter:site" /><meta content="summary" name="twitter:card" /><meta content="silviodonato/cmssw" name="twitter:title" /><meta content="cmssw - CMS Offline Software" name="twitter:description" /><meta content="https://avatars3.githubusercontent.com/u/6177433?v=3&amp;s=400" name="twitter:image:src" />
      <meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="https://avatars3.githubusercontent.com/u/6177433?v=3&amp;s=400" property="og:image" /><meta content="silviodonato/cmssw" property="og:title" /><meta content="https://github.com/silviodonato/cmssw" property="og:url" /><meta content="cmssw - CMS Offline Software" property="og:description" />
      <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">
    <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">
    <link rel="assets" href="https://assets-cdn.github.com/">
    
    <meta name="pjax-timeout" content="1000">
    

    <meta name="msapplication-TileImage" content="/windows-tile.png">
    <meta name="msapplication-TileColor" content="#ffffff">
    <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
    <meta name="google-analytics" content="UA-3769691-2">

<meta content="collector.githubapp.com" name="octolytics-host" /><meta content="collector-cdn.github.com" name="octolytics-script-host" /><meta content="github" name="octolytics-app-id" /><meta content="C1CD4C52:0D19:32CD6A7:5617DC3F" name="octolytics-dimension-request_id" />

<meta content="Rails, view, blob#show" data-pjax-transient="true" name="analytics-event" />


  <meta class="js-ga-set" name="dimension1" content="Logged Out">
    <meta class="js-ga-set" name="dimension4" content="Current repo nav">




    <meta name="is-dotcom" content="true">
        <meta name="hostname" content="github.com">
    <meta name="user-login" content="">

      <link rel="mask-icon" href="https://assets-cdn.github.com/pinned-octocat.svg" color="#4078c0">
      <link rel="icon" type="image/x-icon" href="https://assets-cdn.github.com/favicon.ico">

      <!-- </textarea> --><!-- '"` --><meta content="authenticity_token" name="csrf-param" />
<meta content="zroR+xmwh6VLl0g/NL1QG8oo3tlWI3bEEq0S9unA9YMWYSXmVZD6YFdOZgcl2pep3mVThusqsqFIRiCdhYAewQ==" name="csrf-token" />
    

    <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/github-146afd802a575b0ac3ab74b702dd213fe99b0fdab91c530c4be6777278d548ab.css" media="all" rel="stylesheet" />
    <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/github2-5fbd51630dd27a79b6107d38e57e4c6e8818d18b1d88f4df4bc3f145e211aa82.css" media="all" rel="stylesheet" />
    
    
    


    <meta http-equiv="x-pjax-version" content="de94313abfcb2b2e2f368e2e3020774f">

      
  <meta name="description" content="cmssw - CMS Offline Software">
  <meta name="go-import" content="github.com/silviodonato/cmssw git https://github.com/silviodonato/cmssw.git">

  <meta content="6177433" name="octolytics-dimension-user_id" /><meta content="silviodonato" name="octolytics-dimension-user_login" /><meta content="15160975" name="octolytics-dimension-repository_id" /><meta content="silviodonato/cmssw" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="true" name="octolytics-dimension-repository_is_fork" /><meta content="10969551" name="octolytics-dimension-repository_parent_id" /><meta content="cms-sw/cmssw" name="octolytics-dimension-repository_parent_nwo" /><meta content="10969551" name="octolytics-dimension-repository_network_root_id" /><meta content="cms-sw/cmssw" name="octolytics-dimension-repository_network_root_nwo" />
  <link href="https://github.com/silviodonato/cmssw/commits/BTVHLTDQMOffline_fix.atom" rel="alternate" title="Recent Commits to cmssw:BTVHLTDQMOffline_fix" type="application/atom+xml">

  </head>


  <body class="logged_out   env-production  vis-public fork page-blob">
    <a href="#start-of-content" tabindex="1" class="accessibility-aid js-skip-to-content">Skip to content</a>

    
    
    



      
      <div class="header header-logged-out" role="banner">
  <div class="container clearfix">

    <a class="header-logo-wordmark" href="https://github.com/" data-ga-click="(Logged out) Header, go to homepage, icon:logo-wordmark">
      <span class="mega-octicon octicon-logo-github"></span>
    </a>

    <div class="header-actions" role="navigation">
        <a class="btn btn-primary" href="/join" data-ga-click="(Logged out) Header, clicked Sign up, text:sign-up">Sign up</a>
      <a class="btn" href="/login?return_to=%2Fsilviodonato%2Fcmssw%2Fblob%2FBTVHLTDQMOffline_fix%2FDQMOffline%2FTrigger%2Fsrc%2FBTVHLTOfflineSource.cc" data-ga-click="(Logged out) Header, clicked Sign in, text:sign-in">Sign in</a>
    </div>

    <div class="site-search repo-scope js-site-search" role="search">
      <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/silviodonato/cmssw/search" class="js-site-search-form" data-global-search-url="/search" data-repo-search-url="/silviodonato/cmssw/search" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
  <label class="js-chromeless-input-container form-control">
    <div class="scope-badge">This repository</div>
    <input type="text"
      class="js-site-search-focus js-site-search-field is-clearable chromeless-input"
      data-hotkey="s"
      name="q"
      placeholder="Search"
      aria-label="Search this repository"
      data-global-scope-placeholder="Search GitHub"
      data-repo-scope-placeholder="Search"
      tabindex="1"
      autocapitalize="off">
  </label>
</form>
    </div>

      <ul class="header-nav left" role="navigation">
          <li class="header-nav-item">
            <a class="header-nav-link" href="/explore" data-ga-click="(Logged out) Header, go to explore, text:explore">Explore</a>
          </li>
          <li class="header-nav-item">
            <a class="header-nav-link" href="/features" data-ga-click="(Logged out) Header, go to features, text:features">Features</a>
          </li>
          <li class="header-nav-item">
            <a class="header-nav-link" href="https://enterprise.github.com/" data-ga-click="(Logged out) Header, go to enterprise, text:enterprise">Enterprise</a>
          </li>
          <li class="header-nav-item">
            <a class="header-nav-link" href="/pricing" data-ga-click="(Logged out) Header, go to pricing, text:pricing">Pricing</a>
          </li>
      </ul>

  </div>
</div>



    <div id="start-of-content" class="accessibility-aid"></div>

    <div id="js-flash-container">
</div>


    <div role="main" class="main-content">
        <div itemscope itemtype="http://schema.org/WebPage">
    <div class="pagehead repohead instapaper_ignore readability-menu">

      <div class="container">

        <div class="clearfix">
          

<ul class="pagehead-actions">

  <li>
      <a href="/login?return_to=%2Fsilviodonato%2Fcmssw"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to watch a repository" rel="nofollow">
    <span class="octicon octicon-eye"></span>
    Watch
  </a>
  <a class="social-count" href="/silviodonato/cmssw/watchers">
    1
  </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fsilviodonato%2Fcmssw"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to star a repository" rel="nofollow">
    <span class="octicon octicon-star"></span>
    Star
  </a>

    <a class="social-count js-social-count" href="/silviodonato/cmssw/stargazers">
      0
    </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fsilviodonato%2Fcmssw"
        class="btn btn-sm btn-with-count tooltipped tooltipped-n"
        aria-label="You must be signed in to fork a repository" rel="nofollow">
        <span class="octicon octicon-repo-forked"></span>
        Fork
      </a>

    <a href="/silviodonato/cmssw/network" class="social-count">
      1,586
    </a>
  </li>
</ul>

          <h1 itemscope itemtype="http://data-vocabulary.org/Breadcrumb" class="entry-title public ">
  <span class="mega-octicon octicon-repo-forked"></span>
  <span class="author"><a href="/silviodonato" class="url fn" itemprop="url" rel="author"><span itemprop="title">silviodonato</span></a></span><!--
--><span class="path-divider">/</span><!--
--><strong><a href="/silviodonato/cmssw" data-pjax="#js-repo-pjax-container">cmssw</a></strong>

  <span class="page-context-loader">
    <img alt="" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
  </span>

    <span class="fork-flag">
      <span class="text">forked from <a href="/cms-sw/cmssw">cms-sw/cmssw</a></span>
    </span>
</h1>

        </div>
      </div>
    </div>

    <div class="container">
      <div class="repository-with-sidebar repo-container new-discussion-timeline ">
        <div class="repository-sidebar clearfix">
          
<nav class="sunken-menu repo-nav js-repo-nav js-sidenav-container-pjax js-octicon-loaders"
     role="navigation"
     data-pjax="#js-repo-pjax-container"
     data-issue-count-url="/silviodonato/cmssw/issues/counts">
  <ul class="sunken-menu-group">
    <li class="tooltipped tooltipped-w" aria-label="Code">
      <a href="/silviodonato/cmssw/tree/BTVHLTDQMOffline_fix" aria-label="Code" aria-selected="true" class="js-selected-navigation-item selected sunken-menu-item" data-hotkey="g c" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches /silviodonato/cmssw/tree/BTVHLTDQMOffline_fix">
        <span class="octicon octicon-code"></span> <span class="full-word">Code</span>
        <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>    </li>


    <li class="tooltipped tooltipped-w" aria-label="Pull requests">
      <a href="/silviodonato/cmssw/pulls" aria-label="Pull requests" class="js-selected-navigation-item sunken-menu-item" data-hotkey="g p" data-selected-links="repo_pulls /silviodonato/cmssw/pulls">
          <span class="octicon octicon-git-pull-request"></span> <span class="full-word">Pull requests</span>
          <span class="js-pull-replace-counter"></span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>    </li>

  </ul>
  <div class="sunken-menu-separator"></div>
  <ul class="sunken-menu-group">

    <li class="tooltipped tooltipped-w" aria-label="Pulse">
      <a href="/silviodonato/cmssw/pulse" aria-label="Pulse" class="js-selected-navigation-item sunken-menu-item" data-selected-links="pulse /silviodonato/cmssw/pulse">
        <span class="octicon octicon-pulse"></span> <span class="full-word">Pulse</span>
        <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>    </li>

    <li class="tooltipped tooltipped-w" aria-label="Graphs">
      <a href="/silviodonato/cmssw/graphs" aria-label="Graphs" class="js-selected-navigation-item sunken-menu-item" data-selected-links="repo_graphs repo_contributors /silviodonato/cmssw/graphs">
        <span class="octicon octicon-graph"></span> <span class="full-word">Graphs</span>
        <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>    </li>
  </ul>


</nav>

            <div class="only-with-full-nav">
                
<div class="js-clone-url clone-url open"
  data-protocol-type="http">
  <h3 class="text-small"><span class="text-emphasized">HTTPS</span> clone URL</h3>
  <div class="input-group js-zeroclipboard-container">
    <input type="text" class="input-mini text-small input-monospace js-url-field js-zeroclipboard-target"
           value="https://github.com/silviodonato/cmssw.git" readonly="readonly" aria-label="HTTPS clone URL">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard btn btn-sm zeroclipboard-button tooltipped tooltipped-s" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  
<div class="js-clone-url clone-url "
  data-protocol-type="subversion">
  <h3 class="text-small"><span class="text-emphasized">Subversion</span> checkout URL</h3>
  <div class="input-group js-zeroclipboard-container">
    <input type="text" class="input-mini text-small input-monospace js-url-field js-zeroclipboard-target"
           value="https://github.com/silviodonato/cmssw" readonly="readonly" aria-label="Subversion checkout URL">
    <span class="input-group-button">
      <button aria-label="Copy to clipboard" class="js-zeroclipboard btn btn-sm zeroclipboard-button tooltipped tooltipped-s" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>



<div class="clone-options text-small">You can clone with
  <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/users/set_protocol?protocol_selector=http&amp;protocol_type=clone" class="inline-form js-clone-selector-form " data-form-nonce="97426788f1ff062246f083fae09838f7d7c7293f" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="Ebf5lnLGSj656oLfkD2+DHKwa4SjgE3Gq7fi7TFUsYuH8Ruj8M0+hNvl36utzgAl/q/xNAB4CnPsb8BG54Qzkw==" /></div><button class="btn-link js-clone-selector" data-protocol="http" type="submit">HTTPS</button></form> or <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="/users/set_protocol?protocol_selector=subversion&amp;protocol_type=clone" class="inline-form js-clone-selector-form " data-form-nonce="97426788f1ff062246f083fae09838f7d7c7293f" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /><input name="authenticity_token" type="hidden" value="EBBQygNhvkpH1QD3Vpa+YHnTRlkZJma+6KpjhoKQ15e3V7VSB2EvcTCPhUl5mGi4j9JqJ0LJWnndvyi39gP5aA==" /></div><button class="btn-link js-clone-selector" data-protocol="subversion" type="submit">Subversion</button></form>.
  <a href="https://help.github.com/articles/which-remote-url-should-i-use" class="help tooltipped tooltipped-n" aria-label="Get help on which URL is right for you.">
    <span class="octicon octicon-question"></span>
  </a>
</div>

              <a href="/silviodonato/cmssw/archive/BTVHLTDQMOffline_fix.zip"
                 class="btn btn-sm sidebar-button"
                 aria-label="Download the contents of silviodonato/cmssw as a zip file"
                 title="Download the contents of silviodonato/cmssw as a zip file"
                 rel="nofollow">
                <span class="octicon octicon-cloud-download"></span>
                Download ZIP
              </a>
            </div>
        </div>
        <div id="js-repo-pjax-container" class="repository-content context-loader-container" data-pjax-container>

          

<a href="/silviodonato/cmssw/blob/d967a861ff4f145eb49cd041ccab115955e329cd/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc" class="hidden js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:5b0847774e3a27a7b7674ab630493335 -->

  <div class="file-navigation js-zeroclipboard-container">
    
<div class="select-menu js-menu-container js-select-menu left">
  <span class="btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w"
    title="BTVHLTDQMOffline_fix"
    role="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
    <i>Branch:</i>
    <span class="js-select-button css-truncate-target">BTVHLTDQMOffli…</span>
  </span>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax aria-hidden="true">

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <span class="select-menu-title">Switch branches/tags</span>
        <span class="octicon octicon-x js-menu-close" role="button" aria-label="Close"></span>
      </div>

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Filter branches/tags" id="context-commitish-filter-field" class="js-filterable-field js-navigation-enable" placeholder="Filter branches/tags">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" data-filter-placeholder="Filter branches/tags" class="js-select-menu-tab" role="tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" data-filter-placeholder="Find a tag…" class="js-select-menu-tab" role="tab">Tags</a>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches" role="menu">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/BTVHLTDQMOffline/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="BTVHLTDQMOffline"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="BTVHLTDQMOffline">
                BTVHLTDQMOffline
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/BTVHLTDQMOffline_74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="BTVHLTDQMOffline_74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="BTVHLTDQMOffline_74X">
                BTVHLTDQMOffline_74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/BTVHLTDQMOffline_75X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="BTVHLTDQMOffline_75X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="BTVHLTDQMOffline_75X">
                BTVHLTDQMOffline_75X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open selected"
               href="/silviodonato/cmssw/blob/BTVHLTDQMOffline_fix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="BTVHLTDQMOffline_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="BTVHLTDQMOffline_fix">
                BTVHLTDQMOffline_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/BTVHLTDQMOfflinev2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="BTVHLTDQMOfflinev2"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="BTVHLTDQMOfflinev2">
                BTVHLTDQMOfflinev2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_4_1_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_4_1_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_4_1_X">
                CMSSW_4_1_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_4_4_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_4_4_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_4_4_X">
                CMSSW_4_4_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_5_2_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_5_2_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_5_2_X">
                CMSSW_5_2_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_5_3_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_5_3_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_5_3_X">
                CMSSW_5_3_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_6_1_X_SLHC/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_6_1_X_SLHC"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_6_1_X_SLHC">
                CMSSW_6_1_X_SLHC
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_6_2_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_6_2_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_6_2_X">
                CMSSW_6_2_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_6_2_X_SLHC/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_6_2_X_SLHC"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_6_2_X_SLHC">
                CMSSW_6_2_X_SLHC
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_0_0_pre/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_0_0_pre"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_0_0_pre">
                CMSSW_7_0_0_pre
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_0_0_pre13/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_0_0_pre13"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_0_0_pre13">
                CMSSW_7_0_0_pre13
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_0_BOOSTIO_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_0_BOOSTIO_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_0_BOOSTIO_X">
                CMSSW_7_0_BOOSTIO_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_0_GEANT10_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_0_GEANT10_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_0_GEANT10_X">
                CMSSW_7_0_GEANT10_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_0_ROOT6_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_0_ROOT6_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_0_ROOT6_X">
                CMSSW_7_0_ROOT6_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_0_THREADED_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_0_THREADED_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_0_THREADED_X">
                CMSSW_7_0_THREADED_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_0_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_0_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_0_X">
                CMSSW_7_0_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/CMSSW_7_5_X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="CMSSW_7_5_X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="CMSSW_7_5_X">
                CMSSW_7_5_X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQM_PFMETXX/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQM_PFMETXX"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQM_PFMETXX">
                DQM_PFMETXX
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQM-higgs-btag-fix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQM-higgs-btag-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQM-higgs-btag-fix">
                DQM-higgs-btag-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQM-higgs-btag-fix-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQM-higgs-btag-fix-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQM-higgs-btag-fix-74X">
                DQM-higgs-btag-fix-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQMHiggs_PFMET120/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQMHiggs_PFMET120"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQMHiggs_PFMET120">
                DQMHiggs_PFMET120
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQMIDTight/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQMIDTight"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQMIDTight">
                DQMIDTight
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQMWHbb/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQMWHbb"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQMWHbb">
                DQMWHbb
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQMWHbb_74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQMWHbb_74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQMWHbb_74X">
                DQMWHbb_74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQMWHbb_75X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQMWHbb_75X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQMWHbb_75X">
                DQMWHbb_75X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQMbtagfix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQMbtagfix"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQMbtagfix">
                DQMbtagfix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/DQMbtagfix-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="DQMbtagfix-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="DQMbtagfix-74X">
                DQMbtagfix-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/EtaRangePFSelector/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="EtaRangePFSelector"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="EtaRangePFSelector">
                EtaRangePFSelector
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/HLT2PhotonMET/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="HLT2PhotonMET"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="HLT2PhotonMET">
                HLT2PhotonMET
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/HLT-btag-val-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="HLT-btag-val-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="HLT-btag-val-74X">
                HLT-btag-val-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/HLT-val-btag-fix-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="HLT-val-btag-fix-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="HLT-val-btag-fix-74X">
                HLT-val-btag-fix-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/HLTJetTagWithMatching/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="HLTJetTagWithMatching"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="HLTJetTagWithMatching">
                HLTJetTagWithMatching
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/HLTJetTagWithMatching_73X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="HLTJetTagWithMatching_73X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="HLTJetTagWithMatching_73X">
                HLTJetTagWithMatching_73X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/MT-HLTmumutkVtxProducer/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="MT-HLTmumutkVtxProducer"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="MT-HLTmumutkVtxProducer">
                MT-HLTmumutkVtxProducer
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/MTbtag/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="MTbtag"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="MTbtag">
                MTbtag
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/MTbtag_backup/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="MTbtag_backup"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="MTbtag_backup">
                MTbtag_backup
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/NtuplerHLTdata2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="NtuplerHLTdata2"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="NtuplerHLTdata2">
                NtuplerHLTdata2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/PUinfo/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="PUinfo"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="PUinfo">
                PUinfo
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/PixelJetPuIdMT-75X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="PixelJetPuIdMT-75X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="PixelJetPuIdMT-75X">
                PixelJetPuIdMT-75X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/PixelPUIDMT/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="PixelPUIDMT"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="PixelPUIDMT">
                PixelPUIDMT
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/RemovePUClean/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="RemovePUClean"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="RemovePUClean">
                RemovePUClean
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/RemovePileUpDominatedEvents/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="RemovePileUpDominatedEvents"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="RemovePileUpDominatedEvents">
                RemovePileUpDominatedEvents
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/RemovePileUpDominatedEvents-73X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="RemovePileUpDominatedEvents-73X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="RemovePileUpDominatedEvents-73X">
                RemovePileUpDominatedEvents-73X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/VBFHbb/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="VBFHbb"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="VBFHbb">
                VBFHbb
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/VBFHbb73X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="VBFHbb73X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="VBFHbb73X">
                VBFHbb73X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/VBFHbb_73X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="VBFHbb_73X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="VBFHbb_73X">
                VBFHbb_73X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/VBFHbbDQM/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="VBFHbbDQM"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="VBFHbbDQM">
                VBFHbbDQM
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/VBFHbbfix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="VBFHbbfix"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="VBFHbbfix">
                VBFHbbfix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/VBFHbbfix-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="VBFHbbfix-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="VBFHbbfix-74X">
                VBFHbbfix-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/btag-hlt/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="btag-hlt"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="btag-hlt">
                btag-hlt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/dqm4b/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="dqm4b"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="dqm4b">
                dqm4b
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/dqm4b4/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="dqm4b4"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="dqm4b4">
                dqm4b4
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/dqm4b5/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="dqm4b5"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="dqm4b5">
                dqm4b5
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/dqm4b6/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="dqm4b6"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="dqm4b6">
                dqm4b6
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/dqm4b_rebase/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="dqm4b_rebase"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="dqm4b_rebase">
                dqm4b_rebase
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/etaSelector/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="etaSelector"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="etaSelector">
                etaSelector
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/externaldecay-update-on-top-off-5_3_14/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="externaldecay-update-on-top-off-5_3_14"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="externaldecay-update-on-top-off-5_3_14">
                externaldecay-update-on-top-off-5_3_14
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/fastpv-ptweight/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="fastpv-ptweight"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="fastpv-ptweight">
                fastpv-ptweight
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/fix-val-HLT-btag/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="fix-val-HLT-btag"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="fix-val-HLT-btag">
                fix-val-HLT-btag
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/fix-val-HLT-btag-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="fix-val-HLT-btag-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="fix-val-HLT-btag-74X">
                fix-val-HLT-btag-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/fixedFastPV/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="fixedFastPV"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="fixedFastPV">
                fixedFastPV
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/forCate/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="forCate"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="forCate">
                forCate
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/from-CMSSW_7_1_0_pre9/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="from-CMSSW_7_1_0_pre9"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="from-CMSSW_7_1_0_pre9">
                from-CMSSW_7_1_0_pre9
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/from-CMSSW_7_4_10_patch2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="from-CMSSW_7_4_10_patch2"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="from-CMSSW_7_4_10_patch2">
                from-CMSSW_7_4_10_patch2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/gh-pages/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="gh-pages"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="gh-pages">
                gh-pages
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hlt-btag-up/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hlt-btag-up"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hlt-btag-up">
                hlt-btag-up
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hlt-btag-validation/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hlt-btag-validation"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hlt-btag-validation">
                hlt-btag-validation
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hlt-btag-validation-720pre8/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hlt-btag-validation-720pre8"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hlt-btag-validation-720pre8">
                hlt-btag-validation-720pre8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hlt-btag-validation-no-exception/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hlt-btag-validation-no-exception"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hlt-btag-validation-no-exception">
                hlt-btag-validation-no-exception
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hlt-l1-ntuple/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hlt-l1-ntuple"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hlt-l1-ntuple">
                hlt-l1-ntuple
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hlt-validation-2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hlt-validation-2"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hlt-validation-2">
                hlt-validation-2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hltbtagval-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hltbtagval-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hltbtagval-74X">
                hltbtagval-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/hltbtagval-75X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="hltbtagval-75X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="hltbtagval-75X">
                hltbtagval-75X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/imported-CVS-HEAD/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="imported-CVS-HEAD"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="imported-CVS-HEAD">
                imported-CVS-HEAD
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/l1-ntuple/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="l1-ntuple"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="l1-ntuple">
                l1-ntuple
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/l1-ntuple-74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="l1-ntuple-74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="l1-ntuple-74X">
                l1-ntuple-74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/my_fastPV/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="my_fastPV"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="my_fastPV">
                my_fastPV
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/myHeppy74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="myHeppy74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="myHeppy74X">
                myHeppy74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/mySTEAM/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="mySTEAM"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="mySTEAM">
                mySTEAM
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/myVHbb/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="myVHbb"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="myVHbb">
                myVHbb
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/new-fastPV/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="new-fastPV"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="new-fastPV">
                new-fastPV
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/new_fastPV/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="new_fastPV"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="new_fastPV">
                new_fastPV
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/new_fastPV3/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="new_fastPV3"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="new_fastPV3">
                new_fastPV3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/newTriggerTable/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="newTriggerTable"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="newTriggerTable">
                newTriggerTable
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/newTriggerTableHeppy/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="newTriggerTableHeppy"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="newTriggerTableHeppy">
                newTriggerTableHeppy
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/ntupler-hlt/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="ntupler-hlt"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="ntupler-hlt">
                ntupler-hlt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/patch-1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="patch-1"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="patch-1">
                patch-1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/patch-2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="patch-2"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="patch-2">
                patch-2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/patch-3/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="patch-3"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="patch-3">
                patch-3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/patch-4/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="patch-4"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="patch-4">
                patch-4
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/patch-5/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="patch-5"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="patch-5">
                patch-5
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/ptHatInHltTree/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="ptHatInHltTree"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="ptHatInHltTree">
                ptHatInHltTree
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/runOnData/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="runOnData"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="runOnData">
                runOnData
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/runOnData2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="runOnData2"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="runOnData2">
                runOnData2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/silviodonato/hlt-validation-2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="silviodonato/hlt-validation-2"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="silviodonato/hlt-validation-2">
                silviodonato/hlt-validation-2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/simPrimaryVertex/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="simPrimaryVertex"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="simPrimaryVertex">
                simPrimaryVertex
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/simVertex/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="simVertex"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="simVertex">
                simVertex
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/validation-HLT-btag-fix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="validation-HLT-btag-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="validation-HLT-btag-fix">
                validation-HLT-btag-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/vhbbHeppy74X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="vhbbHeppy74X"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="vhbbHeppy74X">
                vhbbHeppy74X
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/silviodonato/cmssw/blob/vhbbHeppy743/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
               data-name="vhbbHeppy743"
               data-skip-pjax="true"
               rel="nofollow">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <span class="select-menu-item-text css-truncate-target" title="vhbbHeppy743">
                vhbbHeppy743
              </span>
            </a>
        </div>

          <div class="select-menu-no-results">Nothing to show</div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/wreece_111110/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="wreece_111110"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="wreece_111110">wreece_111110</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/untagged-215a6cfabb6564af9a73/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="untagged-215a6cfabb6564af9a73"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="untagged-215a6cfabb6564af9a73">untagged-215a6cfabb6564af9a73</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/michalis_beamspot_44x/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="michalis_beamspot_44x"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="michalis_beamspot_44x">michalis_beamspot_44x</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/merge_44x_and_52x/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="merge_44x_and_52x"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="merge_44x_and_52x">merge_44x_and_52x</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/lucieg_Ap18_44X/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="lucieg_Ap18_44X"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="lucieg_Ap18_44X">lucieg_Ap18_44X</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/joseJan30_BV4_2_1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="joseJan30_BV4_2_1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="joseJan30_BV4_2_1">joseJan30_BV4_2_1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/joseFeb3_BV2_4_1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="joseFeb3_BV2_4_1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="joseFeb3_BV2_4_1">joseFeb3_BV2_4_1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/joseFeb1_BV2_4_1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="joseFeb1_BV2_4_1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="joseFeb1_BV2_4_1">joseFeb1_BV2_4_1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/joseBV2_4_1_limits/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="joseBV2_4_1_limits"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="joseBV2_4_1_limits">joseBV2_4_1_limits</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_11/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_11"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_11">hbbsubstructDev_11</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_10/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_10"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_10">hbbsubstructDev_10</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_9/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_9"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_9">hbbsubstructDev_9</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_8/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_8"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_8">hbbsubstructDev_8</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_7/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_7"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_7">hbbsubstructDev_7</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_6/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_6"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_6">hbbsubstructDev_6</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_4/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_4"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_4">hbbsubstructDev_4</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_3/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_3"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_3">hbbsubstructDev_3</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_2">hbbsubstructDev_2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/hbbsubstructDev_1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="hbbsubstructDev_1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="hbbsubstructDev_1">hbbsubstructDev_1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/cbern_joseAug3b_44X_30Aug12/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="cbern_joseAug3b_44X_30Aug12"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="cbern_joseAug3b_44X_30Aug12">cbern_joseAug3b_44X_30Aug12</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/cbern_backport_04Oct12/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="cbern_backport_04Oct12"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="cbern_backport_04Oct12">cbern_backport_04Oct12</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/cbern_BJ29_cutparser_29Jun12/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="cbern_BJ29_cutparser_29Jun12"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="cbern_BJ29_cutparser_29Jun12">cbern_BJ29_cutparser_29Jun12</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/cbern_12Jun13/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="cbern_12Jun13"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="cbern_12Jun13">cbern_12Jun13</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/benitezj_triggerMatching_23Jan12/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="benitezj_triggerMatching_23Jan12"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="benitezj_triggerMatching_23Jan12">benitezj_triggerMatching_23Jan12</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/benitezj_BV2_4_1_flatntp/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="benitezj_BV2_4_1_flatntp"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="benitezj_BV2_4_1_flatntp">benitezj_BV2_4_1_flatntp</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V7/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V7"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V7">abis_428p7_250_V7</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V6_2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V6_2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V6_2">abis_428p7_250_V6_2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V6_1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V6_1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V6_1">abis_428p7_250_V6_1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V6/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V6"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V6">abis_428p7_250_V6</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V5/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V5"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V5">abis_428p7_250_V5</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V4/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V4"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V4">abis_428p7_250_V4</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V3_1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V3_1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V3_1">abis_428p7_250_V3_1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/abis_428p7_250_V3/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="abis_428p7_250_V3"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="abis_428p7_250_V3">abis_428p7_250_V3</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Wmass_May17_v3/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Wmass_May17_v3"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Wmass_May17_v3">Wmass_May17_v3</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Wmass_May17_v2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Wmass_May17_v2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Wmass_May17_v2">Wmass_May17_v2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/VHbbPartialRecipe2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="VHbbPartialRecipe2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="VHbbPartialRecipe2">VHbbPartialRecipe2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/VHbbPartialRecipe/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="VHbbPartialRecipe"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="VHbbPartialRecipe">VHbbPartialRecipe</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/VHbbFirstImport/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="VHbbFirstImport"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="VHbbFirstImport">VHbbFirstImport</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/VHSept15_AR1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="VHSept15_AR1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="VHSept15_AR1">VHSept15_AR1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/VHNtupleV9_AR1/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="VHNtupleV9_AR1"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="VHNtupleV9_AR1">VHNtupleV9_AR1</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/VHBB_EDMNtupleV3/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="VHBB_EDMNtupleV3"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="VHBB_EDMNtupleV3">VHBB_EDMNtupleV3</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/V21emuCand/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="V21emuCand"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="V21emuCand">V21emuCand</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/V21TauCand_0/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="V21TauCand_0"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="V21TauCand_0">V21TauCand_0</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Sept19th2011_2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Sept19th2011_2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Sept19th2011_2">Sept19th2011_2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jun21th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jun21th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jun21th2011">Jun21th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jun16th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jun16th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jun16th2011">Jun16th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jun15th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jun15th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jun15th2011">Jun15th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jun14th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jun14th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jun14th2011">Jun14th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jun9th2011v2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jun9th2011v2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jun9th2011v2">Jun9th2011v2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jun9th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jun9th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jun9th2011">Jun9th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul28th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul28th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul28th2011">Jul28th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul26th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul26th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul26th2011">Jul26th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul25th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul25th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul25th2011">Jul25th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul22nd2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul22nd2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul22nd2011">Jul22nd2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul21st2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul21st2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul21st2011">Jul21st2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul20th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul20th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul20th2011">Jul20th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul18th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul18th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul18th2011">Jul18th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Jul17th2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Jul17th2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Jul17th2011">Jul17th2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/HBB_EDMNtupleV1_ProcV2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="HBB_EDMNtupleV1_ProcV2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="HBB_EDMNtupleV1_ProcV2">HBB_EDMNtupleV1_ProcV2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV21Mar30/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV21Mar30"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV21Mar30">EdmV21Mar30</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV21Apr10/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV21Apr10"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV21Apr10">EdmV21Apr10</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV21Apr06/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV21Apr06"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV21Apr06">EdmV21Apr06</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV21Apr04/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV21Apr04"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV21Apr04">EdmV21Apr04</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV21Apr03/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV21Apr03"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV21Apr03">EdmV21Apr03</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV21Apr2/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV21Apr2"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV21Apr2">EdmV21Apr2</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV20Mar12/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV20Mar12"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV20Mar12">EdmV20Mar12</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/EdmV9Sept2011/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="EdmV9Sept2011"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="EdmV9Sept2011">EdmV9Sept2011</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/Colin_Nov16_FinalPorting/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="Colin_Nov16_FinalPorting"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="Colin_Nov16_FinalPorting">Colin_Nov16_FinalPorting</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-05-03-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-05-03-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-05-03-2300">CMSSW_7_5_X_2015-05-03-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-05-03-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-05-03-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-05-03-1100">CMSSW_7_5_X_2015-05-03-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-05-02-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-05-02-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-05-02-2300">CMSSW_7_5_X_2015-05-02-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-05-02-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-05-02-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-05-02-1100">CMSSW_7_5_X_2015-05-02-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-05-01-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-05-01-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-05-01-2300">CMSSW_7_5_X_2015-05-01-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-05-01-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-05-01-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-05-01-1100">CMSSW_7_5_X_2015-05-01-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-30-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-30-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-30-2300">CMSSW_7_5_X_2015-04-30-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-30-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-30-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-30-1100">CMSSW_7_5_X_2015-04-30-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-29-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-29-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-29-2300">CMSSW_7_5_X_2015-04-29-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-29-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-29-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-29-1100">CMSSW_7_5_X_2015-04-29-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-28-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-28-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-28-2300">CMSSW_7_5_X_2015-04-28-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-28-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-28-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-28-1100">CMSSW_7_5_X_2015-04-28-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-27-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-27-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-27-2300">CMSSW_7_5_X_2015-04-27-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-27-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-27-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-27-1100">CMSSW_7_5_X_2015-04-27-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-26-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-26-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-26-2300">CMSSW_7_5_X_2015-04-26-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-26-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-26-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-26-1100">CMSSW_7_5_X_2015-04-26-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-25-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-25-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-25-2300">CMSSW_7_5_X_2015-04-25-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-25-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-25-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-25-1100">CMSSW_7_5_X_2015-04-25-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-24-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-24-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-24-2300">CMSSW_7_5_X_2015-04-24-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-24-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-24-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-24-1100">CMSSW_7_5_X_2015-04-24-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-23-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-23-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-23-2300">CMSSW_7_5_X_2015-04-23-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-23-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-23-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-23-1100">CMSSW_7_5_X_2015-04-23-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-22-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-22-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-22-2300">CMSSW_7_5_X_2015-04-22-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-22-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-22-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-22-1100">CMSSW_7_5_X_2015-04-22-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-21-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-21-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-21-2300">CMSSW_7_5_X_2015-04-21-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-21-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-21-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-21-1100">CMSSW_7_5_X_2015-04-21-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-20-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-20-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-20-2300">CMSSW_7_5_X_2015-04-20-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-20-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-20-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-20-1100">CMSSW_7_5_X_2015-04-20-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-19-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-19-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-19-2300">CMSSW_7_5_X_2015-04-19-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-19-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-19-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-19-1100">CMSSW_7_5_X_2015-04-19-1100</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-18-2300/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-18-2300"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-18-2300">CMSSW_7_5_X_2015-04-18-2300</a>
            </div>
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/silviodonato/cmssw/tree/CMSSW_7_5_X_2015-04-18-1100/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc"
                 data-name="CMSSW_7_5_X_2015-04-18-1100"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text css-truncate-target"
                 title="CMSSW_7_5_X_2015-04-18-1100">CMSSW_7_5_X_2015-04-18-1100</a>
            </div>
        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>

    </div>
  </div>
</div>

    <div class="btn-group right">
      <a href="/silviodonato/cmssw/find/BTVHLTDQMOffline_fix"
            class="js-show-file-finder btn btn-sm empty-icon tooltipped tooltipped-nw"
            data-pjax
            data-hotkey="t"
            aria-label="Quickly jump between files">
        <span class="octicon octicon-list-unordered"></span>
      </a>
      <button aria-label="Copy file path to clipboard" class="js-zeroclipboard btn btn-sm zeroclipboard-button tooltipped tooltipped-s" data-copied-hint="Copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </div>

    <div class="breadcrumb js-zeroclipboard-target">
      <span class="repo-root js-repo-root"><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/silviodonato/cmssw/tree/BTVHLTDQMOffline_fix" class="" data-branch="BTVHLTDQMOffline_fix" data-pjax="true" itemscope="url"><span itemprop="title">cmssw</span></a></span></span><span class="separator">/</span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/silviodonato/cmssw/tree/BTVHLTDQMOffline_fix/DQMOffline" class="" data-branch="BTVHLTDQMOffline_fix" data-pjax="true" itemscope="url"><span itemprop="title">DQMOffline</span></a></span><span class="separator">/</span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/silviodonato/cmssw/tree/BTVHLTDQMOffline_fix/DQMOffline/Trigger" class="" data-branch="BTVHLTDQMOffline_fix" data-pjax="true" itemscope="url"><span itemprop="title">Trigger</span></a></span><span class="separator">/</span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/silviodonato/cmssw/tree/BTVHLTDQMOffline_fix/DQMOffline/Trigger/src" class="" data-branch="BTVHLTDQMOffline_fix" data-pjax="true" itemscope="url"><span itemprop="title">src</span></a></span><span class="separator">/</span><strong class="final-path">BTVHLTOfflineSource.cc</strong>
    </div>
  </div>


  <div class="commit file-history-tease">
    <div class="file-history-tease-header">
        <img alt="@silviodonato" class="avatar" height="24" src="https://avatars0.githubusercontent.com/u/6177433?v=3&amp;s=48" width="24" />
        <span class="author"><a href="/silviodonato" rel="author">silviodonato</a></span>
        <time datetime="2015-10-09T15:09:09Z" is="relative-time">Oct 9, 2015</time>
        <div class="commit-title">
            <a href="/silviodonato/cmssw/commit/43aa20e8fbf2156e0ac0dcc1906660d5d219ef4c" class="message" data-pjax="true" title="code safe for missing collections">code safe for missing collections</a>
        </div>
    </div>

    <div class="participation">
      <p class="quickstat">
        <a href="#blob_contributors_box" rel="facebox">
          <strong>1</strong>
           contributor
        </a>
      </p>
      
    </div>
    <div id="blob_contributors_box" style="display:none">
      <h2 class="facebox-header" data-facebox-id="facebox-header">Users who have contributed to this file</h2>
      <ul class="facebox-user-list" data-facebox-id="facebox-description">
          <li class="facebox-user-list-item">
            <img alt="@silviodonato" height="24" src="https://avatars0.githubusercontent.com/u/6177433?v=3&amp;s=48" width="24" />
            <a href="/silviodonato">silviodonato</a>
          </li>
      </ul>
    </div>
  </div>

<div class="file">
  <div class="file-header">
  <div class="file-actions">

    <div class="btn-group">
      <a href="/silviodonato/cmssw/raw/BTVHLTDQMOffline_fix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc" class="btn btn-sm " id="raw-url">Raw</a>
        <a href="/silviodonato/cmssw/blame/BTVHLTDQMOffline_fix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc" class="btn btn-sm js-update-url-with-hash">Blame</a>
      <a href="/silviodonato/cmssw/commits/BTVHLTDQMOffline_fix/DQMOffline/Trigger/src/BTVHLTOfflineSource.cc" class="btn btn-sm " rel="nofollow">History</a>
    </div>


        <button type="button" class="octicon-btn disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <span class="octicon octicon-pencil"></span>
        </button>
        <button type="button" class="octicon-btn octicon-btn-danger disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <span class="octicon octicon-trashcan"></span>
        </button>
  </div>

  <div class="file-info">
      277 lines (221 sloc)
      <span class="file-info-divider"></span>
    11.4 KB
  </div>
</div>

  

  <div class="blob-wrapper data type-c">
      <table class="highlight tab-size js-file-line-container" data-tab-size="8">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DQMOffline/Trigger/interface/BTVHLTOfflineSource.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/Common/interface/TriggerNames.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/Framework/interface/EDAnalyzer.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/Framework/interface/Run.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/Framework/interface/MakerMacros.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/Framework/interface/ESHandle.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/ParameterSet/interface/ParameterSet.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/ServiceRegistry/interface/Service.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>FWCore/MessageLogger/interface/MessageLogger.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DataFormats/Common/interface/Handle.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DataFormats/Common/interface/TriggerResults.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DataFormats/HLTReco/interface/TriggerEvent.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DataFormats/HLTReco/interface/TriggerObject.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DataFormats/HLTReco/interface/TriggerTypeDefs.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>HLTrigger/HLTcore/interface/HLTConfigProvider.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DQMServices/Core/interface/MonitorElement.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>CommonTools/UtilAlgos/interface/DeltaR.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>DataFormats/VertexReco/interface/Vertex.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>TMath.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>TH1F.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>TH2F.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>TProfile.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>TPRegexp.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&lt;</span>cmath<span class="pl-pds">&gt;</span></span></td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line"><span class="pl-k">using</span> <span class="pl-k">namespace</span> <span class="pl-en">edm</span><span class="pl-k">;</span></td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line"><span class="pl-k">using</span> <span class="pl-k">namespace</span> <span class="pl-en">reco</span><span class="pl-k">;</span></td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line"><span class="pl-k">using</span> <span class="pl-k">namespace</span> <span class="pl-en">std</span><span class="pl-k">;</span></td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line"><span class="pl-k">using</span> <span class="pl-k">namespace</span> <span class="pl-en">trigger</span><span class="pl-k">;</span></td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line"><span class="pl-en">BTVHLTOfflineSource::BTVHLTOfflineSource</span>(<span class="pl-k">const</span> edm::ParameterSet&amp; iConfig)</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">{</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">LogDebug</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>BTVHLTOfflineSource<span class="pl-pds">&quot;</span></span>) &lt;&lt; <span class="pl-s"><span class="pl-pds">&quot;</span>constructor....<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">  dirname_                = iConfig.<span class="pl-c1">getUntrackedParameter</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>dirname<span class="pl-pds">&quot;</span></span>,<span class="pl-c1">std::string</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>HLT/BTV/<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">  processname_            = iConfig.<span class="pl-smi">getParameter</span>&lt;std::string&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>processname<span class="pl-pds">&quot;</span></span>);</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">  verbose_                = iConfig.<span class="pl-smi">getUntrackedParameter</span>&lt; <span class="pl-k">bool</span> &gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>verbose<span class="pl-pds">&quot;</span></span>, <span class="pl-c1">false</span>);</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">  triggerSummaryLabel_    = iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>triggerSummaryLabel<span class="pl-pds">&quot;</span></span>);</td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">  triggerResultsLabel_    = iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>triggerResultsLabel<span class="pl-pds">&quot;</span></span>);</td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">  triggerSummaryToken     = consumes &lt;trigger::TriggerEvent&gt; (triggerSummaryLabel_);</td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">  triggerResultsToken     = consumes &lt;edm::TriggerResults&gt;   (triggerResultsLabel_);</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">  triggerSummaryFUToken   = consumes &lt;trigger::TriggerEvent&gt; (<span class="pl-c1">edm::InputTag</span>(triggerSummaryLabel_.<span class="pl-c1">label</span>(),triggerSummaryLabel_.<span class="pl-c1">instance</span>(),<span class="pl-c1">std::string</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>FU<span class="pl-pds">&quot;</span></span>)));</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">  triggerResultsFUToken   = consumes &lt;edm::TriggerResults&gt;   (<span class="pl-c1">edm::InputTag</span>(triggerResultsLabel_.<span class="pl-c1">label</span>(),triggerResultsLabel_.<span class="pl-c1">instance</span>(),<span class="pl-c1">std::string</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>FU<span class="pl-pds">&quot;</span></span>)));</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">  csvCaloTagsToken_       = consumes&lt;reco::JetTagCollection&gt; (<span class="pl-c1">edm::InputTag</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>hltCombinedSecondaryVertexBJetTagsCalo<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">  csvPfTagsToken_         = consumes&lt;reco::JetTagCollection&gt; (<span class="pl-c1">edm::InputTag</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>hltCombinedSecondaryVertexBJetTagsPF<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">  offlineCSVTokenPF_      = consumes&lt;reco::JetTagCollection&gt; (iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>offlineCSVLabelPF<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">  offlineCSVTokenCalo_    = consumes&lt;reco::JetTagCollection&gt; (iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>offlineCSVLabelCalo<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">  hltFastPVToken_         = consumes&lt;std::vector&lt;reco::Vertex&gt; &gt; (iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>hltFastPVLabel<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">  hltPFPVToken_           = consumes&lt;std::vector&lt;reco::Vertex&gt; &gt; (iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>hltPFPVLabel<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">  hltCaloPVToken_         = consumes&lt;std::vector&lt;reco::Vertex&gt; &gt; (iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>hltCaloPVLabel<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">  offlinePVToken_         = consumes&lt;std::vector&lt;reco::Vertex&gt; &gt; (iConfig.<span class="pl-smi">getParameter</span>&lt;edm::InputTag&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>offlinePVLabel<span class="pl-pds">&quot;</span></span>));</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line"> </td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">  std::vector&lt;edm::ParameterSet&gt; paths =  iConfig.<span class="pl-smi">getParameter</span>&lt;std::vector&lt;edm::ParameterSet&gt; &gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>pathPairs<span class="pl-pds">&quot;</span></span>);</td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">for</span>(std::vector&lt;edm::ParameterSet&gt;::iterator <span class="pl-c1">pathconf</span> = paths.<span class="pl-c1">begin</span>() ; <span class="pl-c1">pathconf</span> != paths.<span class="pl-c1">end</span>();  <span class="pl-c1">pathconf</span>++) { </td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">    custompathnamepairs_.<span class="pl-c1">push_back</span>(<span class="pl-c1">make_pair</span>(</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">					     <span class="pl-c1">pathconf</span>-&gt;getParameter&lt;std::string&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>pathname<span class="pl-pds">&quot;</span></span>),</td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">					     <span class="pl-c1">pathconf</span>-&gt;getParameter&lt;std::string&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>pathtype<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">					     ));}</td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line"><span class="pl-en">BTVHLTOfflineSource::~BTVHLTOfflineSource</span>()</td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">{ </td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line"><span class="pl-k">void</span> <span class="pl-en">BTVHLTOfflineSource::dqmBeginRun</span>(<span class="pl-k">const</span> edm::Run&amp; run, <span class="pl-k">const</span> edm::EventSetup&amp; c)</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">{</td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">bool</span> <span class="pl-smi">changed</span>(<span class="pl-c1">true</span>);</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> (!hltConfig_.<span class="pl-c1">init</span>(run, c, processname_, changed)) {</td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">LogDebug</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>BTVHLTOfflineSource<span class="pl-pds">&quot;</span></span>) &lt;&lt; <span class="pl-s"><span class="pl-pds">&quot;</span>HLTConfigProvider failed to initialize.<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">const</span> <span class="pl-k">unsigned</span> <span class="pl-k">int</span> <span class="pl-smi">numberOfPaths</span>(hltConfig_.<span class="pl-c1">size</span>());</td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">for</span>(<span class="pl-k">unsigned</span> <span class="pl-k">int</span> i=<span class="pl-c1">0</span>; i!=numberOfPaths; ++i){</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">    pathname_      = hltConfig_.<span class="pl-c1">triggerName</span>(i);</td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">    filtername_    = <span class="pl-s"><span class="pl-pds">&quot;</span>dummy<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">unsigned</span> <span class="pl-k">int</span> usedPrescale = <span class="pl-c1">1</span>;</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">unsigned</span> <span class="pl-k">int</span> objectType = <span class="pl-c1">0</span>;</td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">    std::string triggerType = <span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">bool</span> trigSelected = <span class="pl-c1">false</span>;</td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> (std::vector&lt;std::pair&lt;std::string, std::string&gt; &gt;::iterator custompathnamepair = custompathnamepairs_.<span class="pl-c1">begin</span>(); </td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">          custompathnamepair != custompathnamepairs_.<span class="pl-c1">end</span>(); ++custompathnamepair){</td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">       <span class="pl-k">if</span>(pathname_.<span class="pl-c1">find</span>(custompathnamepair-&gt;first)!=std::string::npos) { trigSelected = <span class="pl-c1">true</span>; triggerType = custompathnamepair-&gt;second;}</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">      }</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> (!trigSelected) <span class="pl-k">continue</span>;</td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">    hltPathsAll_.<span class="pl-c1">push_back</span>(<span class="pl-c1">PathInfo</span>(usedPrescale, pathname_, <span class="pl-s"><span class="pl-pds">&quot;</span>dummy<span class="pl-pds">&quot;</span></span>, processname_, objectType, triggerType)); </td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">   }</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line"><span class="pl-k">void</span></td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line"><span class="pl-en">BTVHLTOfflineSource::analyze</span>(<span class="pl-k">const</span> edm::Event&amp; iEvent, <span class="pl-k">const</span> edm::EventSetup&amp; iSetup)</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">{ </td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">  iEvent.<span class="pl-c1">getByToken</span>(triggerResultsToken, triggerResults_);</td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">if</span>(!triggerResults_.<span class="pl-c1">isValid</span>()) {</td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">    iEvent.<span class="pl-c1">getByToken</span>(triggerResultsFUToken,triggerResults_);</td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span>(!triggerResults_.<span class="pl-c1">isValid</span>()) {</td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">      <span class="pl-c1">edm::LogInfo</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>BTVHLTOfflineSource<span class="pl-pds">&quot;</span></span>) &lt;&lt; <span class="pl-s"><span class="pl-pds">&quot;</span>TriggerResults not found, <span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">	<span class="pl-s"><span class="pl-pds">&quot;</span>skipping event<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">return</span>;</td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">  }</td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line">  triggerNames_ = iEvent.<span class="pl-c1">triggerNames</span>(*triggerResults_);</td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">  iEvent.<span class="pl-c1">getByToken</span>(triggerSummaryToken,triggerObj_);</td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">if</span>(!triggerObj_.<span class="pl-c1">isValid</span>()) {</td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">    iEvent.<span class="pl-c1">getByToken</span>(triggerSummaryFUToken,triggerObj_);</td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span>(!triggerObj_.<span class="pl-c1">isValid</span>()) {</td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">      <span class="pl-c1">edm::LogInfo</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>BTVHLTOfflineSource<span class="pl-pds">&quot;</span></span>) &lt;&lt; <span class="pl-s"><span class="pl-pds">&quot;</span>TriggerEvent not found, <span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">	<span class="pl-s"><span class="pl-pds">&quot;</span>skipping event<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">return</span>;</td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">  } </td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">  iEvent.<span class="pl-c1">getByToken</span>(csvCaloTagsToken_, csvCaloTags);</td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">  iEvent.<span class="pl-c1">getByToken</span>(csvPfTagsToken_, csvPfTags);</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">Handle</span>&lt;reco::VertexCollection&gt; VertexHandler;</td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">Handle</span>&lt;reco::JetTagCollection&gt; offlineJetTagHandlerPF;</td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">  iEvent.<span class="pl-c1">getByToken</span>(offlineCSVTokenPF_, offlineJetTagHandlerPF);</td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">Handle</span>&lt;reco::JetTagCollection&gt; offlineJetTagHandlerCalo;</td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line">  iEvent.<span class="pl-c1">getByToken</span>(offlineCSVTokenCalo_, offlineJetTagHandlerCalo);</td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">Handle</span>&lt;reco::VertexCollection&gt; offlineVertexHandler;</td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">  iEvent.<span class="pl-c1">getByToken</span>(offlinePVToken_, offlineVertexHandler);</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">if</span>(verbose_ &amp;&amp; iEvent.<span class="pl-c1">id</span>().<span class="pl-c1">event</span>()%<span class="pl-c1">10000</span>==<span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">    cout&lt;&lt;<span class="pl-s"><span class="pl-pds">&quot;</span>Run = <span class="pl-pds">&quot;</span></span>&lt;&lt;iEvent.<span class="pl-c1">id</span>().<span class="pl-c1">run</span>()&lt;&lt;<span class="pl-s"><span class="pl-pds">&quot;</span>, LS = <span class="pl-pds">&quot;</span></span>&lt;&lt;iEvent.<span class="pl-c1">luminosityBlock</span>()&lt;&lt;<span class="pl-s"><span class="pl-pds">&quot;</span>, Event = <span class="pl-pds">&quot;</span></span>&lt;&lt;iEvent.<span class="pl-c1">id</span>().<span class="pl-c1">event</span>()&lt;&lt;endl;  </td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">if</span>(!triggerResults_.<span class="pl-c1">isValid</span>()) <span class="pl-k">return</span>;</td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">   </td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">for</span>(PathInfoCollection::iterator v = hltPathsAll_.<span class="pl-c1">begin</span>(); v!= hltPathsAll_.<span class="pl-c1">end</span>(); ++v ){</td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">unsigned</span> <span class="pl-c1">index</span> = triggerNames_.<span class="pl-c1">triggerIndex</span>(v-&gt;<span class="pl-c1">getPath</span>()); </td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> (<span class="pl-c1">index</span> &lt; triggerNames_.<span class="pl-c1">size</span>() ){     </td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line">     <span class="pl-k">float</span> DR  = <span class="pl-c1">9999</span>.;</td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">     <span class="pl-k">if</span> (csvPfTags.<span class="pl-c1">isValid</span>() &amp;&amp; v-&gt;<span class="pl-c1">getTriggerType</span>() == <span class="pl-s"><span class="pl-pds">&quot;</span>PF<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">     {</td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">auto</span> iter = csvPfTags-&gt;<span class="pl-c1">begin</span>();</td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">float</span> CSV_online = iter-&gt;second;</td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span> (CSV_online&lt;<span class="pl-c1">0</span>) CSV_online = -<span class="pl-c1">0.05</span>;</td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">      v-&gt;<span class="pl-c1">getMEhisto_CSV</span>()-&gt;<span class="pl-c1">Fill</span>(CSV_online);  </td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">      v-&gt;<span class="pl-c1">getMEhisto_Pt</span>()-&gt;<span class="pl-c1">Fill</span>(iter-&gt;first-&gt;<span class="pl-c1">pt</span>()); </td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">      v-&gt;<span class="pl-c1">getMEhisto_Eta</span>()-&gt;<span class="pl-c1">Fill</span>(iter-&gt;first-&gt;<span class="pl-c1">eta</span>());</td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line">      DR  = <span class="pl-c1">9999</span>.;</td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span>(offlineJetTagHandlerPF.<span class="pl-c1">isValid</span>()){</td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line">          <span class="pl-k">for</span> ( reco::JetTagCollection::const_iterator iterO = offlineJetTagHandlerPF-&gt;<span class="pl-c1">begin</span>(); iterO != offlineJetTagHandlerPF-&gt;<span class="pl-c1">end</span>(); iterO++ ){ </td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">float</span> CSV_offline = iterO-&gt;second;</td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> (CSV_offline&lt;<span class="pl-c1">0</span>) CSV_offline = -<span class="pl-c1">0.05</span>;</td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">            DR = <span class="pl-c1">reco::deltaR</span>(iterO-&gt;first-&gt;<span class="pl-c1">eta</span>(),iterO-&gt;first-&gt;<span class="pl-c1">phi</span>(),iter-&gt;first-&gt;<span class="pl-c1">eta</span>(),iter-&gt;first-&gt;<span class="pl-c1">phi</span>());</td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> (DR&lt;<span class="pl-c1">0.3</span>) {</td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line">               v-&gt;<span class="pl-c1">getMEhisto_CSV_RECOvsHLT</span>()-&gt;<span class="pl-c1">Fill</span>(CSV_offline,CSV_online); <span class="pl-k">continue</span>;</td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line">               }</td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line">          }</td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line">      }</td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line">      iEvent.<span class="pl-c1">getByToken</span>(hltPFPVToken_, VertexHandler);</td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span> (VertexHandler.<span class="pl-c1">isValid</span>())</td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line">      { </td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line">        v-&gt;<span class="pl-c1">getMEhisto_PVz</span>()-&gt;<span class="pl-c1">Fill</span>(VertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>()); </td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> (offlineVertexHandler.<span class="pl-c1">isValid</span>()) v-&gt;<span class="pl-c1">getMEhisto_PVz_HLTMinusRECO</span>()-&gt;<span class="pl-c1">Fill</span>(VertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>()-offlineVertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>());</td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line">        }</td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line">      }</td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line">     <span class="pl-k">if</span> (csvCaloTags.<span class="pl-c1">isValid</span>() &amp;&amp; v-&gt;<span class="pl-c1">getTriggerType</span>() == <span class="pl-s"><span class="pl-pds">&quot;</span>Calo<span class="pl-pds">&quot;</span></span>) </td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line">     { </td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">auto</span> iter = csvCaloTags-&gt;<span class="pl-c1">begin</span>();</td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">float</span> CSV_online = iter-&gt;second;</td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span> (CSV_online&lt;<span class="pl-c1">0</span>) CSV_online = -<span class="pl-c1">0.05</span>;</td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line">      v-&gt;<span class="pl-c1">getMEhisto_CSV</span>()-&gt;<span class="pl-c1">Fill</span>(CSV_online);  </td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">      v-&gt;<span class="pl-c1">getMEhisto_Pt</span>()-&gt;<span class="pl-c1">Fill</span>(iter-&gt;first-&gt;<span class="pl-c1">pt</span>()); </td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">      v-&gt;<span class="pl-c1">getMEhisto_Eta</span>()-&gt;<span class="pl-c1">Fill</span>(iter-&gt;first-&gt;<span class="pl-c1">eta</span>());</td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">      DR  = <span class="pl-c1">9999</span>.;</td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span>(offlineJetTagHandlerCalo.<span class="pl-c1">isValid</span>()){</td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line">          <span class="pl-k">for</span> ( reco::JetTagCollection::const_iterator iterO = offlineJetTagHandlerCalo-&gt;<span class="pl-c1">begin</span>(); iterO != offlineJetTagHandlerCalo-&gt;<span class="pl-c1">end</span>(); iterO++ )</td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">          {</td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">float</span> CSV_offline = iterO-&gt;second;</td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> (CSV_offline&lt;<span class="pl-c1">0</span>) CSV_offline = -<span class="pl-c1">0.05</span>;</td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">            DR = <span class="pl-c1">reco::deltaR</span>(iterO-&gt;first-&gt;<span class="pl-c1">eta</span>(),iterO-&gt;first-&gt;<span class="pl-c1">phi</span>(),iter-&gt;first-&gt;<span class="pl-c1">eta</span>(),iter-&gt;first-&gt;<span class="pl-c1">phi</span>());</td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> (DR&lt;<span class="pl-c1">0.3</span>) </td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line">            {</td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">                v-&gt;<span class="pl-c1">getMEhisto_CSV_RECOvsHLT</span>()-&gt;<span class="pl-c1">Fill</span>(CSV_offline,CSV_online); <span class="pl-k">continue</span>;</td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">            }  </td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">          }     </td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">      }</td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">      iEvent.<span class="pl-c1">getByToken</span>(hltFastPVToken_, VertexHandler);</td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span> (VertexHandler.<span class="pl-c1">isValid</span>()) </td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">      {</td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">        v-&gt;<span class="pl-c1">getMEhisto_PVz</span>()-&gt;<span class="pl-c1">Fill</span>(VertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>()); </td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">if</span> (offlineVertexHandler.<span class="pl-c1">isValid</span>()) v-&gt;<span class="pl-c1">getMEhisto_fastPVz_HLTMinusRECO</span>()-&gt;<span class="pl-c1">Fill</span>(VertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>()-offlineVertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>());</td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">       }</td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">      iEvent.<span class="pl-c1">getByToken</span>(hltCaloPVToken_, VertexHandler);</td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span> (VertexHandler.<span class="pl-c1">isValid</span>())</td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">      {</td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line">        v-&gt;<span class="pl-c1">getMEhisto_fastPVz</span>()-&gt;<span class="pl-c1">Fill</span>(VertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>()); </td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">if</span> (offlineVertexHandler.<span class="pl-c1">isValid</span>()) v-&gt;<span class="pl-c1">getMEhisto_PVz_HLTMinusRECO</span>()-&gt;<span class="pl-c1">Fill</span>(VertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>()-offlineVertexHandler-&gt;<span class="pl-c1">begin</span>()-&gt;<span class="pl-c1">z</span>());</td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">       }</td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">       </td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">      }</td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">   }</td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">  </td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line"><span class="pl-k">void</span> </td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line"><span class="pl-en">BTVHLTOfflineSource::bookHistograms</span>(DQMStore::IBooker &amp; iBooker, edm::Run <span class="pl-k">const</span> &amp; run, edm::EventSetup <span class="pl-k">const</span> &amp; c)</td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">{</td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">  iBooker.<span class="pl-c1">setCurrentFolder</span>(dirname_);</td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">   <span class="pl-k">for</span>(PathInfoCollection::iterator v = hltPathsAll_.<span class="pl-c1">begin</span>(); v!= hltPathsAll_.<span class="pl-c1">end</span>(); ++v ){</td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">     <span class="pl-c">//</span></td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">     std::string trgPathName = <span class="pl-c1">HLTConfigProvider::removeVersion</span>(v-&gt;<span class="pl-c1">getPath</span>());</td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">     std::string subdirName  = dirname_ +<span class="pl-s"><span class="pl-pds">&quot;</span>/<span class="pl-pds">&quot;</span></span>+ trgPathName;</td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line">     std::string trigPath    = <span class="pl-s"><span class="pl-pds">&quot;</span>(<span class="pl-pds">&quot;</span></span>+trgPathName+<span class="pl-s"><span class="pl-pds">&quot;</span>)<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line">     iBooker.<span class="pl-c1">setCurrentFolder</span>(subdirName);  </td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">     std::string <span class="pl-smi">labelname</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>HLT<span class="pl-pds">&quot;</span></span>);</td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">     std::string <span class="pl-smi">histoname</span>(labelname+<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-pds">&quot;</span></span>);</td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">     std::string <span class="pl-smi">title</span>(labelname+<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-pds">&quot;</span></span>);</td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">      </td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">     histoname = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_CSV<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">     title = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_CSV <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">     MonitorElement * CSV =  iBooker.<span class="pl-c1">book1D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">110</span>,-<span class="pl-c1">0.1</span>,<span class="pl-c1">1</span>);</td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">     </td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line">     histoname = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_Pt<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">     title = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_Pt <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">     MonitorElement * Pt =  iBooker.<span class="pl-c1">book1D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">100</span>,<span class="pl-c1">0</span>,<span class="pl-c1">400</span>);</td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">     </td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">     histoname = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_Eta<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">     title = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_Eta <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">     MonitorElement * Eta =  iBooker.<span class="pl-c1">book1D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">60</span>,-<span class="pl-c1">3.0</span>,<span class="pl-c1">3.0</span>);</td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line">     histoname = <span class="pl-s"><span class="pl-pds">&quot;</span>RECOvsHLT_CSV<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">     title = <span class="pl-s"><span class="pl-pds">&quot;</span>offline CSV vs online CSV <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line">     MonitorElement * CSV_RECOvsHLT =  iBooker.<span class="pl-c1">book2D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">110</span>,-<span class="pl-c1">0.1</span>,<span class="pl-c1">1</span>,<span class="pl-c1">110</span>,-<span class="pl-c1">0.1</span>,<span class="pl-c1">1</span>);</td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">     histoname = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_PVz<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">     title = <span class="pl-s"><span class="pl-pds">&quot;</span>online z(PV) <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">     MonitorElement * PVz =  iBooker.<span class="pl-c1">book1D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">80</span>,-<span class="pl-c1">20</span>,<span class="pl-c1">20</span>);</td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">     </td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">     histoname = labelname+<span class="pl-s"><span class="pl-pds">&quot;</span>_fastPVz<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">     title = <span class="pl-s"><span class="pl-pds">&quot;</span>online z(fastPV) <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">     MonitorElement * fastPVz =  iBooker.<span class="pl-c1">book1D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">80</span>,-<span class="pl-c1">20</span>,<span class="pl-c1">20</span>);</td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">     </td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">     histoname = <span class="pl-s"><span class="pl-pds">&quot;</span>HLTMinusRECO_PVz<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">     title = <span class="pl-s"><span class="pl-pds">&quot;</span>online z(PV) - offline z(PV) <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">     MonitorElement * PVz_HLTMinusRECO =  iBooker.<span class="pl-c1">book1D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">200</span>,-<span class="pl-c1">0.5</span>,<span class="pl-c1">0.5</span>);</td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line">     </td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line">     histoname = <span class="pl-s"><span class="pl-pds">&quot;</span>HLTMinusRECO_fastPVz<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line">     title = <span class="pl-s"><span class="pl-pds">&quot;</span>online z(fastPV) - offline z(PV) <span class="pl-pds">&quot;</span></span>+trigPath;</td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line">     MonitorElement * fastPVz_HLTMinusRECO =  iBooker.<span class="pl-c1">book1D</span>(histoname.<span class="pl-c1">c_str</span>(),title.<span class="pl-c1">c_str</span>(),<span class="pl-c1">100</span>,-<span class="pl-c1">2</span>,<span class="pl-c1">2</span>);</td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">     v-&gt;<span class="pl-c1">setHistos</span>(CSV,Pt,Eta,CSV_RECOvsHLT,PVz,fastPVz,PVz_HLTMinusRECO,fastPVz_HLTMinusRECO);  </td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">   }</td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
</table>

  </div>

</div>

<a href="#jump-to-line" rel="facebox[.linejump]" data-hotkey="l" style="display:none">Jump to Line</a>
<div id="jump-to-line" style="display:none">
  <!-- </textarea> --><!-- '"` --><form accept-charset="UTF-8" action="" class="js-jump-to-line-form" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
    <input class="linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
    <button type="submit" class="btn">Go</button>
</form></div>

        </div>
      </div>
      <div class="modal-backdrop"></div>
    </div>
  </div>


    </div>

      <div class="container">
  <div class="site-footer" role="contentinfo">
    <ul class="site-footer-links right">
        <li><a href="https://status.github.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
      <li><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
      <li><a href="https://shop.github.com" data-ga-click="Footer, go to shop, text:shop">Shop</a></li>
        <li><a href="https://github.com/blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a href="https://github.com/about" data-ga-click="Footer, go to about, text:about">About</a></li>
        <li><a href="https://github.com/pricing" data-ga-click="Footer, go to pricing, text:pricing">Pricing</a></li>

    </ul>

    <a href="https://github.com" aria-label="Homepage">
      <span class="mega-octicon octicon-mark-github" title="GitHub"></span>
</a>
    <ul class="site-footer-links">
      <li>&copy; 2015 <span title="0.09124s from github-fe134-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
        <li><a href="https://github.com/site/terms" data-ga-click="Footer, go to terms, text:terms">Terms</a></li>
        <li><a href="https://github.com/site/privacy" data-ga-click="Footer, go to privacy, text:privacy">Privacy</a></li>
        <li><a href="https://github.com/security" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li><a href="https://github.com/contact" data-ga-click="Footer, go to contact, text:contact">Contact</a></li>
        <li><a href="https://help.github.com" data-ga-click="Footer, go to help, text:help">Help</a></li>
    </ul>
  </div>
</div>



    
    
    

    <div id="ajax-error-message" class="flash flash-error">
      <span class="octicon octicon-alert"></span>
      <button type="button" class="flash-close js-flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
        <span class="octicon octicon-x"></span>
      </button>
      Something went wrong with that request. Please try again.
    </div>


      <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/frameworks-f8473dece7242da6a20d52313634881b3975c52cebaa1e6c38157c0f26185691.js"></script>
      <script async="async" crossorigin="anonymous" src="https://assets-cdn.github.com/assets/github-b0c2bfdb75a0cb36101e784febac2830fe49b229b273b8c9ed5ec34768421e8a.js"></script>
      
      
    <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner hidden">
      <span class="octicon octicon-alert"></span>
      <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
      <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
    </div>
  </body>
</html>

