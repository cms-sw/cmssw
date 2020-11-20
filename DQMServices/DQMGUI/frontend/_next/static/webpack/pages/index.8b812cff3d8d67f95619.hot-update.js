webpackHotUpdate_N_E("pages/index",{

/***/ "./components/overlayWithAnotherPlot/index.tsx":
/*!*****************************************************!*\
  !*** ./components/overlayWithAnotherPlot/index.tsx ***!
  \*****************************************************/
/*! exports provided: OverlayWithAnotherPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "OverlayWithAnotherPlot", function() { return OverlayWithAnotherPlot; });
/* harmony import */ var _babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/toConsumableArray */ "./node_modules/@babel/runtime/helpers/esm/toConsumableArray.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd/lib/modal/Modal */ "./node_modules/antd/lib/modal/Modal.js");
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../containers/display/content/folderPath */ "./containers/display/content/folderPath.tsx");
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! clean-deep */ "./node_modules/clean-deep/src/index.js");
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_11___default = /*#__PURE__*/__webpack_require__.n(clean_deep__WEBPACK_IMPORTED_MODULE_11__);
/* harmony import */ var _containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../containers/search/styledComponents */ "./containers/search/styledComponents.tsx");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/overlayWithAnotherPlot/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2__["createElement"];











var OverlayWithAnotherPlot = function OverlayWithAnotherPlot(_ref) {
  _s();

  var visible = _ref.visible,
      setOpenOverlayWithAnotherPlotModal = _ref.setOpenOverlayWithAnotherPlotModal;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({
    folder_path: '',
    name: ''
  }),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState, 2),
      overlaidPlots = _React$useState2[0],
      setOverlaidPlots = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([]),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      folders = _React$useState4[0],
      setFolders = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](''),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState5, 2),
      currentFolder = _React$useState6[0],
      setCurrentFolder = _React$useState6[1];

  var _React$useState7 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({}),
      _React$useState8 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState7, 2),
      plot = _React$useState8[0],
      setPlot = _React$useState8[1];

  var _React$useState9 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](),
      _React$useState10 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState9, 2),
      height = _React$useState10[0],
      setHeight = _React$useState10[1];

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"])();
  var query = router.query;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_2__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__["store"]),
      updated_by_not_older_than = _React$useContext.updated_by_not_older_than;

  var params = {
    dataset_name: query.dataset_name,
    run_number: query.run_number,
    notOlderThan: updated_by_not_older_than,
    folders_path: overlaidPlots.folder_path,
    plot_name: overlaidPlots.name
  };
  var api = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_6__["choose_api"])(params);
  var data_get_by_mount = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"])(api, {}, [overlaidPlots.folder_path]);
  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    var copy = Object(_babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__["default"])(folders);

    var index = folders.indexOf(currentFolder);

    if (index >= 0) {
      var rest = copy.splice(0, index + 1);
      setFolders(rest);
      var joinderFolders = rest.join('/');
      setOverlaidPlots({
        folder_path: joinderFolders,
        name: ''
      });
    } else {
      copy.push(currentFolder); //we're cleaning copy array, because we want to delete empty string. 
      // We need to remove it because when we're joining array with empty string 
      // we're getting a string with '/' in the beginning.

      var cleaned_array = clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(copy) ? clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(copy) : [];
      setFolders(cleaned_array);

      var _joinderFolders = copy.join('/');

      if (cleaned_array.length === 0) {
        setOverlaidPlots({
          folder_path: '',
          name: ''
        });
      }

      setOverlaidPlots({
        folder_path: _joinderFolders,
        name: ''
      });
    }
  }, [currentFolder]);
  var modalRef = react__WEBPACK_IMPORTED_MODULE_2__["useRef"](null);
  var data = data_get_by_mount.data;
  var folders_or_plots = data ? data.data : [];

  var changeFolderPathByBreadcrumb = function changeFolderPathByBreadcrumb(item) {
    // const folders_from_breadcrumb = item.folder_path.split('/') 
    var folders_from_breadcrumb = [];
    var cleaned_folders_array = clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(folders_from_breadcrumb) ? clean_deep__WEBPACK_IMPORTED_MODULE_11___default()(folders_from_breadcrumb) : [];
    setFolders(cleaned_folders_array);

    if (cleaned_folders_array.length > 0) {
      setCurrentFolder(cleaned_folders_array[cleaned_folders_array.length - 1]);
    } else {
      setCurrentFolder('');
    }
  };

  return __jsx(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default.a, {
    visible: visible,
    onCancel: function onCancel() {
      setOpenOverlayWithAnotherPlotModal(false);
      setCurrentFolder('');
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 89,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 96,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 97,
      columnNumber: 9
    }
  }, __jsx(_containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__["FolderPath"], {
    folder_path: overlaidPlots.folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
      columnNumber: 11
    }
  })), !data_get_by_mount.isLoading && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%',
      flex: '1 1 auto'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 102,
      columnNumber: 11
    }
  }, folders_or_plots.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.subdir && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setCurrentFolder(folder_or_plot.subdir);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 107,
        columnNumber: 21
      }
    }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Icon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 108,
        columnNumber: 23
      }
    }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 109,
        columnNumber: 23
      }
    }, folder_or_plot.subdir)));
  })), data_get_by_mount.isLoading && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%',
      display: 'flex',
      justifyContent: 'center',
      height: '100%',
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 118,
      columnNumber: 11
    }
  }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_12__["Spinner"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 119,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 123,
      columnNumber: 11
    }
  }, !data_get_by_mount.isLoading && folders_or_plots.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.name && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 16,
      onClick: function onClick() {
        return setPlot(folder_or_plot);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 128,
        columnNumber: 21
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Button"], {
      block: true,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 129,
        columnNumber: 23
      }
    }, folder_or_plot.name)));
  }))));
};

_s(OverlayWithAnotherPlot, "22XQJ3uLL7mKQo1ldpFWyLcRAOs=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"], _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"]];
});

_c = OverlayWithAnotherPlot;

var _c;

$RefreshReg$(_c, "OverlayWithAnotherPlot");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9vdmVybGF5V2l0aEFub3RoZXJQbG90L2luZGV4LnRzeCJdLCJuYW1lcyI6WyJPdmVybGF5V2l0aEFub3RoZXJQbG90IiwidmlzaWJsZSIsInNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwiLCJSZWFjdCIsImZvbGRlcl9wYXRoIiwibmFtZSIsIm92ZXJsYWlkUGxvdHMiLCJzZXRPdmVybGFpZFBsb3RzIiwiZm9sZGVycyIsInNldEZvbGRlcnMiLCJjdXJyZW50Rm9sZGVyIiwic2V0Q3VycmVudEZvbGRlciIsInBsb3QiLCJzZXRQbG90IiwiaGVpZ2h0Iiwic2V0SGVpZ2h0Iiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJzdG9yZSIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJwYXJhbXMiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwibm90T2xkZXJUaGFuIiwiZm9sZGVyc19wYXRoIiwicGxvdF9uYW1lIiwiYXBpIiwiY2hvb3NlX2FwaSIsImRhdGFfZ2V0X2J5X21vdW50IiwidXNlUmVxdWVzdCIsImNvcHkiLCJpbmRleCIsImluZGV4T2YiLCJyZXN0Iiwic3BsaWNlIiwiam9pbmRlckZvbGRlcnMiLCJqb2luIiwicHVzaCIsImNsZWFuZWRfYXJyYXkiLCJjbGVhbkRlZXAiLCJsZW5ndGgiLCJtb2RhbFJlZiIsImRhdGEiLCJmb2xkZXJzX29yX3Bsb3RzIiwiY2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYiIsIml0ZW0iLCJmb2xkZXJzX2Zyb21fYnJlYWRjcnVtYiIsImNsZWFuZWRfZm9sZGVyc19hcnJheSIsInBhZGRpbmciLCJpc0xvYWRpbmciLCJ3aWR0aCIsImZsZXgiLCJtYXAiLCJmb2xkZXJfb3JfcGxvdCIsInN1YmRpciIsImRpc3BsYXkiLCJqdXN0aWZ5Q29udGVudCIsImFsaWduSXRlbXMiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7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QUFFQyxXQUFLLEVBQUUsTUFBVDtBQUFpQkssYUFBTyxFQUFFLE1BQTFCO0FBQWtDQyxvQkFBYyxFQUFFLFFBQWxEO0FBQTREM0MsWUFBTSxFQUFFLE1BQXBFO0FBQTRFNEMsZ0JBQVUsRUFBRTtBQUF4RixLQUFaO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDRFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQXRCSixFQTJCSSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxDQUFDN0IsaUJBQWlCLENBQUNxQixTQUFuQixJQUFnQ04sZ0JBQWdCLENBQUNTLEdBQWpCLENBQXFCLFVBQUNDLGNBQUQsRUFBeUI7QUFDNUUsV0FDRSw0REFDR0EsY0FBYyxDQUFDakQsSUFBZixJQUNDLE1BQUMsd0NBQUQ7QUFBSyxVQUFJLEVBQUUsRUFBWDtBQUFlLGFBQU8sRUFBRTtBQUFBLGVBQU1RLE9BQU8sQ0FBQ3lDLGNBQUQsQ0FBYjtBQUFBLE9BQXhCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLDJDQUFEO0FBQVEsV0FBSyxNQUFiO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FBZUEsY0FBYyxDQUFDakQsSUFBOUIsQ0FERixDQUZKLENBREY7QUFTRCxHQVYrQixDQURsQyxDQTNCSixDQVBGLENBREY7QUFxREQsQ0F4SE07O0dBQU1MLHNCO1VBT0lpQixxRCxFQWFXYSw0RDs7O0tBcEJmOUIsc0IiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguOGI4MTJjZmYzZDhkNjdmOTU2MTkuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0J1xyXG5pbXBvcnQgTW9kYWwgZnJvbSAnYW50ZC9saWIvbW9kYWwvTW9kYWwnXHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJ1xyXG5cclxuaW1wb3J0IHsgUGFyYW1zRm9yQXBpUHJvcHMsIFBsb3RvdmVybGFpZFNlcGFyYXRlbHlQcm9wcywgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJ1xyXG5pbXBvcnQgeyBJY29uLCBTdHlsZWRBIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnXHJcbmltcG9ydCB7IGNob29zZV9hcGkgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnXHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0J1xyXG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlUmVxdWVzdCdcclxuaW1wb3J0IHsgQnV0dG9uLCBDb2wsIFJvdyB9IGZyb20gJ2FudGQnXHJcbmltcG9ydCB7IEZvbGRlclBhdGggfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvY29udGVudC9mb2xkZXJQYXRoJ1xyXG5pbXBvcnQgeyBQYXJzZWRVcmxRdWVyeUlucHV0IH0gZnJvbSAncXVlcnlzdHJpbmcnXHJcbmltcG9ydCBjbGVhbkRlZXAgZnJvbSAnY2xlYW4tZGVlcCdcclxuaW1wb3J0IHsgU3Bpbm5lciB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvc2VhcmNoL3N0eWxlZENvbXBvbmVudHMnXHJcblxyXG5pbnRlcmZhY2UgT3ZlcmxheVdpdGhBbm90aGVyUGxvdFByb3BzIHtcclxuICB2aXNpYmxlOiBib29sZWFuO1xyXG4gIHNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWw6IGFueVxyXG59XHJcblxyXG5leHBvcnQgY29uc3QgT3ZlcmxheVdpdGhBbm90aGVyUGxvdCA9ICh7IHZpc2libGUsIHNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwgfTogT3ZlcmxheVdpdGhBbm90aGVyUGxvdFByb3BzKSA9PiB7XHJcbiAgY29uc3QgW292ZXJsYWlkUGxvdHMsIHNldE92ZXJsYWlkUGxvdHNdID0gUmVhY3QudXNlU3RhdGU8UGxvdG92ZXJsYWlkU2VwYXJhdGVseVByb3BzPih7IGZvbGRlcl9wYXRoOiAnJywgbmFtZTogJycgfSlcclxuICBjb25zdCBbZm9sZGVycywgc2V0Rm9sZGVyc10gPSBSZWFjdC51c2VTdGF0ZTwoc3RyaW5nIHwgdW5kZWZpbmVkKVtdPihbXSlcclxuICBjb25zdCBbY3VycmVudEZvbGRlciwgc2V0Q3VycmVudEZvbGRlcl0gPSBSZWFjdC51c2VTdGF0ZTxzdHJpbmcgfCB1bmRlZmluZWQ+KCcnKVxyXG4gIGNvbnN0IFtwbG90LCBzZXRQbG90XSA9IFJlYWN0LnVzZVN0YXRlKHt9KVxyXG4gIGNvbnN0IFtoZWlnaHQsIHNldEhlaWdodF0gPSBSZWFjdC51c2VTdGF0ZSgpXHJcblxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG4gIGNvbnN0IHsgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcclxuXHJcbiAgY29uc3QgcGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcyA9IHtcclxuICAgIGRhdGFzZXRfbmFtZTogcXVlcnkuZGF0YXNldF9uYW1lIGFzIHN0cmluZyxcclxuICAgIHJ1bl9udW1iZXI6IHF1ZXJ5LnJ1bl9udW1iZXIgYXMgc3RyaW5nLFxyXG4gICAgbm90T2xkZXJUaGFuOiB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxyXG4gICAgZm9sZGVyc19wYXRoOiBvdmVybGFpZFBsb3RzLmZvbGRlcl9wYXRoLFxyXG4gICAgcGxvdF9uYW1lOiBvdmVybGFpZFBsb3RzLm5hbWVcclxuICB9XHJcblxyXG4gIGNvbnN0IGFwaSA9IGNob29zZV9hcGkocGFyYW1zKVxyXG4gIGNvbnN0IGRhdGFfZ2V0X2J5X21vdW50ID0gdXNlUmVxdWVzdChhcGksXHJcbiAgICB7fSxcclxuICAgIFtvdmVybGFpZFBsb3RzLmZvbGRlcl9wYXRoXVxyXG4gICk7XHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBjb3B5ID0gWy4uLmZvbGRlcnNdXHJcbiAgICBjb25zdCBpbmRleCA9IGZvbGRlcnMuaW5kZXhPZihjdXJyZW50Rm9sZGVyKVxyXG5cclxuICAgIGlmIChpbmRleCA+PSAwKSB7XHJcbiAgICAgIGNvbnN0IHJlc3QgPSBjb3B5LnNwbGljZSgwLCBpbmRleCArIDEpXHJcbiAgICAgIHNldEZvbGRlcnMocmVzdClcclxuICAgICAgY29uc3Qgam9pbmRlckZvbGRlcnMgPSByZXN0LmpvaW4oJy8nKVxyXG4gICAgICBzZXRPdmVybGFpZFBsb3RzKHsgZm9sZGVyX3BhdGg6IGpvaW5kZXJGb2xkZXJzLCBuYW1lOiAnJyB9KVxyXG4gICAgfVxyXG4gICAgZWxzZSB7XHJcbiAgICAgIGNvcHkucHVzaChjdXJyZW50Rm9sZGVyKVxyXG4gICAgICAvL3dlJ3JlIGNsZWFuaW5nIGNvcHkgYXJyYXksIGJlY2F1c2Ugd2Ugd2FudCB0byBkZWxldGUgZW1wdHkgc3RyaW5nLiBcclxuICAgICAgLy8gV2UgbmVlZCB0byByZW1vdmUgaXQgYmVjYXVzZSB3aGVuIHdlJ3JlIGpvaW5pbmcgYXJyYXkgd2l0aCBlbXB0eSBzdHJpbmcgXHJcbiAgICAgIC8vIHdlJ3JlIGdldHRpbmcgYSBzdHJpbmcgd2l0aCAnLycgaW4gdGhlIGJlZ2lubmluZy5cclxuICAgICAgY29uc3QgY2xlYW5lZF9hcnJheSA9IGNsZWFuRGVlcChjb3B5KSA/IGNsZWFuRGVlcChjb3B5KSA6IFtdXHJcbiAgICAgIHNldEZvbGRlcnMoY2xlYW5lZF9hcnJheSlcclxuICAgICAgY29uc3Qgam9pbmRlckZvbGRlcnMgPSBjb3B5LmpvaW4oJy8nKVxyXG4gICAgICBpZiAoY2xlYW5lZF9hcnJheS5sZW5ndGggPT09IDApIHtcclxuICAgICAgICBzZXRPdmVybGFpZFBsb3RzKHsgZm9sZGVyX3BhdGg6ICcnLCBuYW1lOiAnJyB9KVxyXG4gICAgICB9XHJcbiAgICAgIHNldE92ZXJsYWlkUGxvdHMoeyBmb2xkZXJfcGF0aDogam9pbmRlckZvbGRlcnMsIG5hbWU6ICcnIH0pXHJcbiAgICB9XHJcbiAgfSwgW2N1cnJlbnRGb2xkZXJdKVxyXG5cclxuICBjb25zdCBtb2RhbFJlZiA9IFJlYWN0LnVzZVJlZihudWxsKTtcclxuXHJcbiAgY29uc3QgeyBkYXRhIH0gPSBkYXRhX2dldF9ieV9tb3VudFxyXG4gIGNvbnN0IGZvbGRlcnNfb3JfcGxvdHMgPSBkYXRhID8gZGF0YS5kYXRhIDogW11cclxuICBjb25zdCBjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1iID0gKGl0ZW06IFBhcnNlZFVybFF1ZXJ5SW5wdXQpID0+IHtcclxuICAgIC8vIGNvbnN0IGZvbGRlcnNfZnJvbV9icmVhZGNydW1iID0gaXRlbS5mb2xkZXJfcGF0aC5zcGxpdCgnLycpIFxyXG4gICAgY29uc3QgZm9sZGVyc19mcm9tX2JyZWFkY3J1bWIgPSBbXVxyXG4gICAgY29uc3QgY2xlYW5lZF9mb2xkZXJzX2FycmF5ID0gY2xlYW5EZWVwKGZvbGRlcnNfZnJvbV9icmVhZGNydW1iKSA/IGNsZWFuRGVlcChmb2xkZXJzX2Zyb21fYnJlYWRjcnVtYikgOiBbXVxyXG4gICAgc2V0Rm9sZGVycyhjbGVhbmVkX2ZvbGRlcnNfYXJyYXkpXHJcbiAgICBpZiAoY2xlYW5lZF9mb2xkZXJzX2FycmF5Lmxlbmd0aCA+IDApIHtcclxuICAgICAgc2V0Q3VycmVudEZvbGRlcihjbGVhbmVkX2ZvbGRlcnNfYXJyYXlbY2xlYW5lZF9mb2xkZXJzX2FycmF5Lmxlbmd0aCAtIDFdKVxyXG4gICAgfVxyXG4gICAgZWxzZSB7XHJcbiAgICAgIHNldEN1cnJlbnRGb2xkZXIoJycpXHJcbiAgICB9XHJcbiAgfVxyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPE1vZGFsXHJcbiAgICAgIHZpc2libGU9e3Zpc2libGV9XHJcbiAgICAgIG9uQ2FuY2VsPXsoKSA9PiB7XHJcbiAgICAgICAgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbChmYWxzZSlcclxuICAgICAgICBzZXRDdXJyZW50Rm9sZGVyKCcnKVxyXG4gICAgICB9fVxyXG4gICAgPlxyXG4gICAgICA8Um93IGd1dHRlcj17MTZ9ID5cclxuICAgICAgICA8Q29sIHN0eWxlPXt7IHBhZGRpbmc6IDggfX0+XHJcbiAgICAgICAgICA8Rm9sZGVyUGF0aCBmb2xkZXJfcGF0aD17b3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aH0gY2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYj17Y2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYn0gLz5cclxuICAgICAgICA8L0NvbD5cclxuICAgICAgICB7XHJcbiAgICAgICAgICAhZGF0YV9nZXRfYnlfbW91bnQuaXNMb2FkaW5nICYmXHJcbiAgICAgICAgICA8Um93IHN0eWxlPXt7IHdpZHRoOiAnMTAwJScsIGZsZXg6ICcxIDEgYXV0bycgfX0+XHJcbiAgICAgICAgICAgIHtmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3Quc3ViZGlyICYmXHJcbiAgICAgICAgICAgICAgICAgICAgPENvbCBzcGFuPXs4fSBvbkNsaWNrPXsoKSA9PiBzZXRDdXJyZW50Rm9sZGVyKGZvbGRlcl9vcl9wbG90LnN1YmRpcil9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEljb24gLz5cclxuICAgICAgICAgICAgICAgICAgICAgIDxTdHlsZWRBPntmb2xkZXJfb3JfcGxvdC5zdWJkaXJ9PC9TdHlsZWRBPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICA8Lz5cclxuICAgICAgICAgICAgICApXHJcbiAgICAgICAgICAgIH0pfVxyXG4gICAgICAgICAgPC9Sb3c+XHJcbiAgICAgICAgfVxyXG4gICAgICAgIHtkYXRhX2dldF9ieV9tb3VudC5pc0xvYWRpbmcgJiZcclxuICAgICAgICAgIDxSb3cgc3R5bGU9e3sgd2lkdGg6ICcxMDAlJywgZGlzcGxheTogJ2ZsZXgnLCBqdXN0aWZ5Q29udGVudDogJ2NlbnRlcicsIGhlaWdodDogJzEwMCUnLCBhbGlnbkl0ZW1zOiAnY2VudGVyJyB9fT5cclxuICAgICAgICAgICAgPFNwaW5uZXIgLz5cclxuICAgICAgICAgIDwvUm93PlxyXG4gICAgICAgIH1cclxuICAgICAgICB7XHJcbiAgICAgICAgICA8Um93PntcclxuICAgICAgICAgICAgIWRhdGFfZ2V0X2J5X21vdW50LmlzTG9hZGluZyAmJiBmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3QubmFtZSAmJlxyXG4gICAgICAgICAgICAgICAgICAgIDxDb2wgc3Bhbj17MTZ9IG9uQ2xpY2s9eygpID0+IHNldFBsb3QoZm9sZGVyX29yX3Bsb3QpfT5cclxuICAgICAgICAgICAgICAgICAgICAgIDxCdXR0b24gYmxvY2s+e2ZvbGRlcl9vcl9wbG90Lm5hbWV9PC9CdXR0b24+XHJcbiAgICAgICAgICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIDwvPlxyXG4gICAgICAgICAgICAgIClcclxuICAgICAgICAgICAgfSlcclxuICAgICAgICAgIH1cclxuICAgICAgICAgIDwvUm93PlxyXG4gICAgICAgIH1cclxuICAgICAgPC9Sb3c+XHJcbiAgICA8L01vZGFsPlxyXG4gIClcclxufSJdLCJzb3VyY2VSb290IjoiIn0=